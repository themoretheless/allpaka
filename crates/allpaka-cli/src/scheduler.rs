//! Model-aware continuous batching, backpressure, and cancellation.

use std::collections::{BTreeMap, VecDeque};

pub type RequestId = u64;

#[derive(Debug)]
pub struct ScheduledRequest<T> {
    pub id: RequestId,
    pub model: String,
    pub queued_at: u64,
    pub context_tokens: usize,
    pub payload: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueError {
    QueueFull,
    DuplicateRequest,
}

#[derive(Debug)]
pub struct ContinuousBatcher<T> {
    queues: BTreeMap<String, VecDeque<ScheduledRequest<T>>>,
    max_queued: usize,
    max_batch: usize,
    max_batch_context_tokens: usize,
    queued: usize,
}

impl<T> ContinuousBatcher<T> {
    pub fn new(
        max_queued: usize,
        max_batch: usize,
        max_batch_context_tokens: usize,
    ) -> Self {
        Self {
            queues: BTreeMap::new(),
            max_queued,
            max_batch: max_batch.max(1),
            max_batch_context_tokens,
            queued: 0,
        }
    }

    pub fn queued(&self) -> usize {
        self.queued
    }

    pub fn enqueue(&mut self, request: ScheduledRequest<T>) -> Result<(), EnqueueError> {
        if self.queued >= self.max_queued {
            return Err(EnqueueError::QueueFull);
        }
        if self
            .queues
            .values()
            .flatten()
            .any(|queued| queued.id == request.id)
        {
            return Err(EnqueueError::DuplicateRequest);
        }
        self.queues
            .entry(request.model.clone())
            .or_default()
            .push_back(request);
        self.queued += 1;
        Ok(())
    }

    pub fn cancel(&mut self, id: RequestId) -> Option<ScheduledRequest<T>> {
        for queue in self.queues.values_mut() {
            if let Some(index) = queue.iter().position(|request| request.id == id) {
                self.queued -= 1;
                return queue.remove(index);
            }
        }
        None
    }

    pub fn next_batch(&mut self) -> Vec<ScheduledRequest<T>> {
        let model = self
            .queues
            .iter()
            .filter_map(|(model, queue)| queue.front().map(|head| (model.clone(), head.queued_at)))
            .min_by_key(|(_, queued_at)| *queued_at)
            .map(|(model, _)| model);
        let Some(model) = model else {
            return Vec::new();
        };
        let queue = self.queues.get_mut(&model).expect("selected queue");
        let mut batch = Vec::new();
        let mut context_tokens = 0usize;
        while batch.len() < self.max_batch {
            let Some(next) = queue.front() else {
                break;
            };
            if !batch.is_empty()
                && context_tokens.saturating_add(next.context_tokens)
                    > self.max_batch_context_tokens
            {
                break;
            }
            let request = queue.pop_front().expect("front existed");
            context_tokens = context_tokens.saturating_add(request.context_tokens);
            batch.push(request);
            self.queued -= 1;
        }
        if queue.is_empty() {
            self.queues.remove(&model);
        }
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::{ContinuousBatcher, EnqueueError, ScheduledRequest};

    fn request(id: u64, model: &str, queued_at: u64, tokens: usize) -> ScheduledRequest<u64> {
        ScheduledRequest {
            id,
            model: model.into(),
            queued_at,
            context_tokens: tokens,
            payload: id,
        }
    }

    #[test]
    fn batches_are_model_isolated_and_oldest_model_wins() {
        let mut scheduler = ContinuousBatcher::new(8, 4, 100);
        scheduler.enqueue(request(1, "large", 20, 10)).unwrap();
        scheduler.enqueue(request(2, "small", 10, 10)).unwrap();
        scheduler.enqueue(request(3, "small", 30, 10)).unwrap();
        let batch = scheduler.next_batch();
        assert_eq!(batch.iter().map(|r| r.id).collect::<Vec<_>>(), [2, 3]);
        assert!(batch.iter().all(|r| r.model == "small"));
    }

    #[test]
    fn backpressure_cancellation_and_token_budget_are_enforced() {
        let mut scheduler = ContinuousBatcher::new(2, 4, 15);
        scheduler.enqueue(request(1, "m", 1, 10)).unwrap();
        scheduler.enqueue(request(2, "m", 2, 10)).unwrap();
        assert_eq!(
            scheduler.enqueue(request(3, "m", 3, 1)),
            Err(EnqueueError::QueueFull)
        );
        assert_eq!(scheduler.cancel(2).unwrap().payload, 2);
        scheduler.enqueue(request(3, "m", 3, 10)).unwrap();
        assert_eq!(scheduler.next_batch().len(), 1);
        assert_eq!(scheduler.queued(), 1);
    }
}
