//! Cooperative cancellation primitive shared across Studio background jobs.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::watch;

struct Inner {
    tx: watch::Sender<bool>,
    requested: AtomicBool,
    closed: AtomicBool,
}

/// Cloneable cooperative cancellation token backed by `tokio::sync::watch`.
///
/// Unifies job cancellation across `experiments`, `matrix_jobs`, `judge_runs`,
/// and `background` without polling loops or disparate channel types.
#[derive(Clone)]
pub(crate) struct CancelToken {
    inner: Arc<Inner>,
    rx: watch::Receiver<bool>,
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    pub(crate) fn new() -> Self {
        let (tx, rx) = watch::channel(false);
        Self {
            inner: Arc::new(Inner {
                tx,
                requested: AtomicBool::new(false),
                closed: AtomicBool::new(false),
            }),
            rx,
        }
    }

    pub(crate) fn closed() -> Self {
        let token = Self::new();
        token.close();
        token
    }

    /// Request cancellation once while the job is still open.
    /// Returns `true` only on the first cancellation request before `close()`.
    pub(crate) fn cancel(&self) -> bool {
        if self.inner.closed.load(Ordering::Acquire) {
            return false;
        }
        let first = !self.inner.requested.swap(true, Ordering::AcqRel);
        let _ = self.inner.tx.send(true);
        first
    }

    /// Request cancellation and return `true` as long as the job is still open.
    pub(crate) fn cancel_active(&self) -> bool {
        if self.inner.closed.load(Ordering::Acquire) {
            return false;
        }
        self.inner.requested.store(true, Ordering::Release);
        let _ = self.inner.tx.send(true);
        true
    }

    /// Mark the job as finished so late `cancel()` calls report `false`.
    pub(crate) fn close(&self) {
        self.inner.closed.store(true, Ordering::Release);
    }

    /// Non-blocking check whether cancellation has been requested.
    #[inline]
    pub(crate) fn is_cancelled(&self) -> bool {
        self.inner.requested.load(Ordering::Acquire) || *self.rx.borrow()
    }

    /// Resolves immediately if already cancelled, or as soon as `cancel()` fires.
    pub(crate) async fn cancelled(&mut self) {
        if self.is_cancelled() {
            return;
        }
        while self.rx.changed().await.is_ok() {
            if *self.rx.borrow() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CancelToken;

    #[tokio::test]
    async fn cancel_wakes_waiters_and_reports_first_request() {
        let token = CancelToken::new();
        let mut waiter = token.clone();
        let handle = tokio::spawn(async move {
            waiter.cancelled().await;
            waiter.is_cancelled()
        });
        assert!(!token.is_cancelled());
        assert!(token.cancel());
        assert!(!token.cancel());
        assert!(token.cancel_active());
        assert!(handle.await.unwrap());
        token.close();
        assert!(!token.cancel_active());
    }
}
