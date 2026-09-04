use crate::model_registry::{ModelRegistry, RegistryError};
use crate::scheduler::{ContinuousBatcher, EnqueueError, RequestId, ScheduledRequest};
use allpaka_model::prefix_cache::PrefixCache;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct ServingLimits {
    pub max_queued: usize,
    pub max_batch: usize,
    pub max_batch_context_tokens: usize,
    pub model_budget_bytes: u64,
    pub memory_budget_bytes: u64,
    pub prefix_budget_bytes: usize,
}

impl Default for ServingLimits {
    fn default() -> Self {
        Self {
            max_queued: 256,
            max_batch: 16,
            max_batch_context_tokens: 32 * 1024,
            model_budget_bytes: u64::MAX,
            memory_budget_bytes: u64::MAX,
            prefix_budget_bytes: 512 << 20,
        }
    }
}

pub struct ServingRuntime<M, P, V> {
    queue: ContinuousBatcher<P>,
    models: ModelRegistry<M>,
    prefixes: PrefixCache<V>,
    next_request_id: RequestId,
    queue_clock: u64,
}

impl<M, P, V> ServingRuntime<M, P, V> {
    pub fn new(limits: ServingLimits) -> Self {
        Self {
            queue: ContinuousBatcher::new(
                limits.max_queued,
                limits.max_batch,
                limits.max_batch_context_tokens,
            ),
            models: ModelRegistry::new(limits.model_budget_bytes),
            prefixes: PrefixCache::new(limits.prefix_budget_bytes),
            next_request_id: 1,
            queue_clock: 0,
        }
    }

    pub fn install_model(
        &mut self,
        name: String,
        model: M,
        bytes: u64,
    ) -> Result<(), RegistryError> {
        self.models.install(name, model, bytes)
    }

    pub fn model(&mut self, name: &str) -> Option<Arc<M>> {
        self.models.get(name)
    }

    pub fn model_names(&self) -> Vec<String> {
        self.models.model_names().map(ToOwned::to_owned).collect()
    }

    pub fn enqueue(
        &mut self,
        model: String,
        context_tokens: usize,
        payload: P,
    ) -> Result<RequestId, EnqueueError> {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        self.queue_clock = self.queue_clock.wrapping_add(1);
        self.queue.enqueue(ScheduledRequest {
            id,
            model,
            context_tokens,
            queued_at: self.queue_clock,
            payload,
        })?;
        Ok(id)
    }

    pub fn next_batch(&mut self) -> Vec<ScheduledRequest<P>> {
        self.queue.next_batch()
    }

    pub fn queued(&self) -> usize {
        self.queue.queued()
    }

    pub fn resident_model_bytes(&self) -> u64 {
        self.models.resident_bytes()
    }

    pub fn prefixes(&mut self) -> &mut PrefixCache<V> {
        &mut self.prefixes
    }
}

#[cfg(test)]
mod tests {
    use super::{ServingLimits, ServingRuntime};

    #[test]
    fn admission_batches_requests_for_an_installed_model() {
        let mut runtime = ServingRuntime::<u32, &'static str, Vec<u8>>::new(ServingLimits {
            max_queued: 4,
            max_batch: 2,
            max_batch_context_tokens: 16,
            model_budget_bytes: 64,
            memory_budget_bytes: u64::MAX,
            prefix_budget_bytes: 64,
        });
        runtime.install_model("target".into(), 7, 32).unwrap();
        let lease = runtime.model("target").unwrap();
        let first = runtime.enqueue("target".into(), 6, "one").unwrap();
        let second = runtime.enqueue("target".into(), 7, "two").unwrap();

        let batch = runtime.next_batch();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, first);
        assert_eq!(batch[1].id, second);
        assert_eq!(*lease, 7);
        assert_eq!(runtime.queued(), 0);
        assert_eq!(runtime.resident_model_bytes(), 32);
        let _ = runtime.prefixes();
    }
}
