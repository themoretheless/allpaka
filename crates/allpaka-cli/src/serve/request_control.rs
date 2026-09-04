//! Request lifetime independent of the single inference owner.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interrupted { Cancelled, Deadline }

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { Self::Cancelled => "request cancelled", Self::Deadline => "request deadline exceeded" })
    }
}
impl std::error::Error for Interrupted {}

#[derive(Debug)]
struct State {
    cancelled: AtomicBool,
    streaming: AtomicBool,
    deadline: Instant,
}

#[derive(Default)]
pub struct Registry {
    entries: Mutex<HashMap<String, Weak<State>>>,
    next_id: AtomicU64,
}

pub struct RequestControl {
    pub id: String,
    state: Arc<State>,
    registry: Arc<Registry>,
}

impl Registry {
    pub fn register(self: &Arc<Self>, requested: Option<&str>, timeout: Duration) -> anyhow::Result<RequestControl> {
        let id = requested.map(str::to_owned).unwrap_or_else(|| {
            format!("req-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
        });
        anyhow::ensure!(!id.is_empty() && id.len() <= 128 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'), "invalid request_id");
        let mut entries = self.entries.lock().map_err(|_| anyhow::anyhow!("request registry poisoned"))?;
        anyhow::ensure!(!entries.contains_key(&id), "request_id already active");
        let state = Arc::new(State {
            cancelled: AtomicBool::new(false),
            streaming: AtomicBool::new(false),
            deadline: Instant::now() + timeout,
        });
        entries.insert(id.clone(), Arc::downgrade(&state));
        Ok(RequestControl { id, state, registry: Arc::clone(self) })
    }

    pub fn cancel(&self, id: &str) -> bool {
        let Ok(entries) = self.entries.lock() else { return false };
        let Some(state) = entries.get(id).and_then(Weak::upgrade) else { return false };
        state.cancelled.store(true, Ordering::Release);
        true
    }
}

impl RequestControl {
    pub fn check(&self) -> Result<(), Interrupted> {
        if self.state.cancelled.load(Ordering::Acquire) { return Err(Interrupted::Cancelled) }
        if Instant::now() >= self.state.deadline { return Err(Interrupted::Deadline) }
        Ok(())
    }
    pub fn start_stream(&self) { self.state.streaming.store(true, Ordering::Release); }
    pub fn streaming(&self) -> bool { self.state.streaming.load(Ordering::Acquire) }
}

impl Drop for RequestControl {
    fn drop(&mut self) {
        if let Ok(mut entries) = self.registry.entries.lock() { entries.remove(&self.id); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_visible_across_threads_and_drop_releases_id() {
        let registry = Arc::new(Registry::default());
        let control = registry.register(Some("test"), Duration::from_secs(5)).unwrap();
        assert!(registry.register(Some("test"), Duration::from_secs(5)).is_err());
        let other = Arc::clone(&registry);
        std::thread::spawn(move || assert!(other.cancel("test"))).join().unwrap();
        assert_eq!(control.check(), Err(Interrupted::Cancelled));
        drop(control);
        assert!(!registry.cancel("test"));
        assert!(registry.register(Some("test"), Duration::from_secs(5)).is_ok());
    }

    #[test]
    fn deadline_includes_time_spent_waiting_in_queue() {
        let registry = Arc::new(Registry::default());
        let control = registry.register(None, Duration::ZERO).unwrap();
        assert_eq!(control.check(), Err(Interrupted::Deadline));
    }
}
