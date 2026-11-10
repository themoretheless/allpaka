//! One cancellation signal shared by the long-running evaluation jobs.
use std::sync::Arc;
use tokio::sync::watch;

/// A cloneable, one-way stop request: once cancelled it stays cancelled.
/// Jobs either poll [`CancelToken::is_cancelled`] between steps or race
/// [`CancelToken::cancelled`] against the step in flight.
#[derive(Clone, Debug)]
pub(crate) struct CancelToken(Arc<watch::Sender<bool>>);

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    pub(crate) fn new() -> Self {
        Self(Arc::new(watch::Sender::new(false)))
    }
    /// Request cancellation. Returns true only for the call that changed the
    /// token, so a caller can tell a first request from a repeated one.
    pub(crate) fn cancel(&self) -> bool {
        self.0
            .send_if_modified(|cancelled| !std::mem::replace(cancelled, true))
    }
    pub(crate) fn is_cancelled(&self) -> bool {
        *self.0.borrow()
    }
    /// Resolve once cancellation has been requested, immediately if it
    /// already has been.
    pub(crate) async fn cancelled(&self) {
        let mut receiver = self.0.subscribe();
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
