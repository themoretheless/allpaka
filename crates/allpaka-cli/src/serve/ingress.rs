//! Bounded HTTP readers keep control requests reachable during GPU work.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, TrySendError};
use std::time::Duration;

pub(super) fn start(
    listener: TcpListener,
    default_model: String,
    limits: ServingLimits,
) -> Receiver<(String, usize, QueuedConnection)> {
    let (sender, receiver) = mpsc::sync_channel(limits.max_queued);
    let registry = Arc::new(RequestRegistry::default());
    let readers = Arc::new(AtomicUsize::new(0));
    std::thread::spawn(move || {
        for incoming in listener.incoming() {
            let Ok(mut stream) = incoming else { break };
            if readers.fetch_add(1, Ordering::AcqRel) >= 16 {
                readers.fetch_sub(1, Ordering::AcqRel);
                let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                let _ = respond(&mut stream, 429, &json!({"error":"HTTP reader capacity exhausted"}));
                continue;
            }
            let readers = Arc::clone(&readers);
            let sender = sender.clone();
            let registry = Arc::clone(&registry);
            let model = default_model.clone();
            std::thread::spawn(move || {
                struct ReaderLease(Arc<AtomicUsize>);
                impl Drop for ReaderLease {
                    fn drop(&mut self) { self.0.fetch_sub(1, Ordering::AcqRel); }
                }
                let _lease = ReaderLease(readers);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
                let mut error_stream = stream.try_clone().ok();
                match prepare_connection(stream, &model, limits.max_batch_context_tokens, &registry) {
                    Ok(Some(prepared)) => match sender.try_send(prepared) {
                        Ok(()) => {},
                        Err(TrySendError::Full((_, _, mut request))) => {
                            let _ = respond(&mut request.stream, 429, &json!({"error":"inference queue full"}));
                        },
                        Err(TrySendError::Disconnected((_, _, mut request))) => {
                            let _ = respond(&mut request.stream, 503, &json!({"error":"inference worker unavailable"}));
                        },
                    },
                    Ok(None) => {},
                    Err(error) => {
                        if let Some(ref mut stream) = error_stream {
                            let _ = respond(stream, 400, &json!({"error":error.to_string()}));
                        }
                    },
                }
            });
        }
    });
    receiver
}
