//! Bounded HTTP reader worker pool keeps control requests reachable during GPU work
//! without spawning a new OS thread per TCP connection.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, TrySendError};
use std::time::Duration;

const READER_WORKERS: usize = 16;

pub(super) fn start(
    listener: TcpListener,
    default_model: String,
    limits: ServingLimits,
) -> Receiver<(String, usize, QueuedConnection)> {
    let (sender, receiver) = mpsc::sync_channel(limits.max_queued);
    let (conn_tx, conn_rx) = mpsc::sync_channel::<TcpStream>(READER_WORKERS);
    let conn_rx = Arc::new(Mutex::new(conn_rx));
    let registry = Arc::new(RequestRegistry::default());
    let active_readers = Arc::new(AtomicUsize::new(0));

    for _ in 0..READER_WORKERS {
        let conn_rx = Arc::clone(&conn_rx);
        let active_readers = Arc::clone(&active_readers);
        let sender = sender.clone();
        let registry = Arc::clone(&registry);
        let model = default_model.clone();
        let max_context_tokens = limits.max_batch_context_tokens;
        std::thread::spawn(move || loop {
            let stream = {
                let Ok(guard) = conn_rx.lock() else { break };
                match guard.recv() {
                    Ok(stream) => stream,
                    Err(_) => break,
                }
            };
            struct ReaderLease<'a>(&'a AtomicUsize);
            impl Drop for ReaderLease<'_> {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::AcqRel);
                }
            }
            let _lease = ReaderLease(&active_readers);
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
            let mut error_stream = stream.try_clone().ok();
            match prepare_connection(stream, &model, max_context_tokens, &registry) {
                Ok(Some(prepared)) => match sender.try_send(prepared) {
                    Ok(()) => {}
                    Err(TrySendError::Full((_, _, mut request))) => {
                        let _ = respond(
                            &mut request.stream,
                            429,
                            &json!({"error":"inference queue full"}),
                        );
                    }
                    Err(TrySendError::Disconnected((_, _, mut request))) => {
                        let _ = respond(
                            &mut request.stream,
                            503,
                            &json!({"error":"inference worker unavailable"}),
                        );
                    }
                },
                Ok(None) => {}
                Err(error) => {
                    if let Some(ref mut stream) = error_stream {
                        let _ = respond(stream, 400, &json!({"error":error.to_string()}));
                    }
                }
            }
        });
    }

    std::thread::spawn(move || {
        for incoming in listener.incoming() {
            let Ok(mut stream) = incoming else { break };
            if active_readers.fetch_add(1, Ordering::AcqRel) >= READER_WORKERS {
                active_readers.fetch_sub(1, Ordering::AcqRel);
                let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                let _ = respond(
                    &mut stream,
                    429,
                    &json!({"error":"HTTP reader capacity exhausted"}),
                );
                continue;
            }
            if let Err(err) = conn_tx.try_send(stream) {
                active_readers.fetch_sub(1, Ordering::AcqRel);
                let mut stream = match err {
                    TrySendError::Full(s) | TrySendError::Disconnected(s) => s,
                };
                let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                let _ = respond(
                    &mut stream,
                    429,
                    &json!({"error":"HTTP reader capacity exhausted"}),
                );
            }
        }
    });
    receiver
}
