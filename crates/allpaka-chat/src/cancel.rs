//! One cancellation signal shared by the long-running evaluation jobs.
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::watch;

#[derive(Debug)]
struct Inner {
    tx: watch::Sender<bool>,
    /// Set once the job finished, so late cancel requests report `false`.
    closed: AtomicBool,
}

/// A cloneable, one-way stop request: once cancelled it stays cancelled.
/// Jobs either poll [`CancelToken::is_cancelled`] between steps or race
/// [`CancelToken::cancelled`] against the step in flight.
#[derive(Clone, Debug)]
pub(crate) struct CancelToken(Arc<Inner>);

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Inner {
            tx: watch::Sender::new(false),
            closed: AtomicBool::new(false),
        }))
    }
    /// A token for a job that has already finished.
    pub(crate) fn closed() -> Self {
        let token = Self::new();
        token.close();
        token
    }
    /// Request cancellation. Returns true only for the call that changed the
    /// token while the job is still open, so a caller can tell a first
    /// request from a repeated or late one.
    pub(crate) fn cancel(&self) -> bool {
        if self.0.closed.load(Ordering::Acquire) {
            return false;
        }
        self.0
            .tx
            .send_if_modified(|cancelled| !std::mem::replace(cancelled, true))
    }
    /// Request cancellation and return true as long as the job is still open,
    /// even when cancellation was already requested.
    pub(crate) fn cancel_active(&self) -> bool {
        if self.0.closed.load(Ordering::Acquire) {
            return false;
        }
        self.0.tx.send_replace(true);
        true
    }
    /// Mark the job as finished so later cancel calls report `false`.
    pub(crate) fn close(&self) {
        self.0.closed.store(true, Ordering::Release);
    }
    /// Whether both handles belong to the same token.
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn is_cancelled(&self) -> bool {
        *self.0.tx.borrow()
    }
    /// Resolve once cancellation has been requested, immediately if it
    /// already has been.
    pub(crate) async fn cancelled(&self) {
        let mut receiver = self.0.tx.subscribe();
        // The sender lives as long as `self`, so waiting cannot fail.
        let _ = receiver.wait_for(|cancelled| *cancelled).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn only_the_first_cancel_reports_a_change() {
        let token = CancelToken::new();
        let clone = token.clone();
        assert!(!token.is_cancelled());
        assert!(clone.cancel());
        assert!(!token.cancel());
        assert!(token.is_cancelled() && clone.is_cancelled());
    }

    #[test]
    fn closed_tokens_reject_late_requests() {
        let token = CancelToken::new();
        assert!(token.cancel());
        assert!(token.cancel_active());
        token.close();
        assert!(!token.cancel_active());
        assert!(!token.cancel());
        assert!(!CancelToken::closed().cancel());
    }

    #[tokio::test]
    async fn cancelled_resolves_before_and_after_the_request() {
        let token = CancelToken::new();
        let waiter = tokio::spawn({
            let token = token.clone();
            async move { token.cancelled().await }
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        token.cancel();
        tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("a pending waiter wakes on cancel")
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), token.cancelled())
            .await
            .expect("an already cancelled token resolves at once");
    }
}
