//! Two long generations taking turns, one step each.
//!
//! A short request waiting behind a long one is served in between
//! (`preemptible`). Two long ones cannot be: whichever ran second would wait
//! for all of the first. So a second long chat request runs on a worker
//! thread in a session of its own, and the two generations pass a turn
//! between them after every step (a prefill chunk, a decode token) through a
//! FIFO ticket gate. Each waits for one step of the other, never for all of
//! it. Switching sessions on the GPU replays parked graphs, so a turn costs
//! about 3% (`tests/cuda_session_switch_cost.rs`).

use super::*;
use std::sync::{Condvar, Mutex as StdMutex};

/// A fair turn: whoever asked first goes next.
pub(super) struct Gate {
    state: StdMutex<(u64, u64)>, // (next ticket to hand out, ticket being served)
    turn: Condvar,
}

impl Gate {
    /// A gate whose first turn (ticket 0) is already held by the caller.
    fn held() -> Self {
        Self {
            state: StdMutex::new((1, 0)),
            turn: Condvar::new(),
        }
    }

    /// Wait for a turn.
    pub(super) fn enter(&self) {
        let mut state = self.state.lock().unwrap();
        let ticket = state.0;
        state.0 += 1;
        while state.1 != ticket {
            state = self.turn.wait(state).unwrap();
        }
    }

    /// End a turn.
    pub(super) fn leave(&self) {
        self.state.lock().unwrap().1 += 1;
        self.turn.notify_all();
    }

    /// End this turn and wait for the next one: anyone who asked meanwhile
    /// goes first.
    pub(super) fn pass(&self) {
        self.leave();
        self.enter();
    }
}

/// The gate shared with the running worker, if one runs.
static ACTIVE: StdMutex<Option<Arc<Gate>>> = StdMutex::new(None);

pub(super) fn active() -> Option<Arc<Gate>> {
    ACTIVE.lock().unwrap().clone()
}

/// What a worker needs to run a generation on its own: the model shared, and
/// what the dispatcher would otherwise take from the locked service.
#[derive(Clone)]
pub(super) struct Shared {
    pub(super) name: String,
    pub(super) model: Arc<Model<'static>>,
    pub(super) tokenizer: Arc<Tokenizer>,
    pub(super) force_think: bool,
    pub(super) budget: memory::Budget,
}

/// Whether a waiting request should run alongside the current one: a chat
/// completion for the same model, without a named session, while no worker
/// runs. (One much shorter than the running request is served in between
/// instead; see `preemptible`.)
pub(super) fn eligible(requested: &str, shared: &Shared, connection: &QueuedConnection) -> bool {
    matches!(
        routes::classify(&connection.request_line).map(|m| m.route),
        Some(Route::ChatCompletions)
    ) && requested == shared.name
        && connection.request.get("session_id").is_none()
        && active().is_none()
}

/// Start `connection` on a worker that takes turns with the caller, and return
/// the gate; the caller holds the first turn and must `pass` after each step
/// and `leave` when its own generation ends.
pub(super) fn spawn(shared: Shared, connection: QueuedConnection) -> Arc<Gate> {
    let gate = Arc::new(Gate::held());
    *ACTIVE.lock().unwrap() = Some(Arc::clone(&gate));
    let worker_gate = Arc::clone(&gate);
    std::thread::spawn(move || {
        worker_gate.enter();
        let mut template = match Template::detect(&shared.tokenizer) {
            Ok(template) => template,
            Err(_) => {
                finish(&worker_gate);
                return;
            }
        };
        if let Template::ChatMl { force_think, .. } = &mut template {
            *force_think = shared.force_think;
        }
        let rag = RagTools::load();
        let pass =
            |_: &Model<'static>, _: &Tokenizer, _: &Template, _: &memory::Budget, _: usize| {
                worker_gate.pass();
            };
        preemptible::serve(
            &shared.name,
            connection,
            &shared.model,
            &shared.tokenizer,
            &template,
            &shared.budget,
            &rag,
            Some(&pass),
        );
        finish(&worker_gate);
    });
    gate
}

/// The worker is done: stop gating, then give up its turn.
fn finish(gate: &Arc<Gate>) {
    let mut active = ACTIVE.lock().unwrap();
    if active.as_ref().is_some_and(|g| Arc::ptr_eq(g, gate)) {
        *active = None;
    }
    drop(active);
    gate.leave();
}

#[cfg(test)]
mod tests {
    use super::Gate;
    use std::sync::{Arc, Mutex};

    /// Two parties passing the turn after every step alternate strictly.
    #[test]
    fn turns_alternate_one_step_each() {
        let gate = Arc::new(Gate::held());
        let log = Arc::new(Mutex::new(Vec::new()));
        let (g, l) = (Arc::clone(&gate), Arc::clone(&log));
        let worker = std::thread::spawn(move || {
            g.enter();
            for i in 0..5 {
                l.lock().unwrap().push(format!("w{i}"));
                g.pass();
            }
            g.leave();
        });
        // Let the worker queue up for its first turn.
        while gate.state.lock().unwrap().0 < 2 {
            std::thread::yield_now();
        }
        for i in 0..5 {
            log.lock().unwrap().push(format!("m{i}"));
            gate.pass();
        }
        gate.leave();
        worker.join().unwrap();
        let log = log.lock().unwrap().clone();
        assert_eq!(
            log,
            ["m0", "w0", "m1", "w1", "m2", "w2", "m3", "w3", "m4", "w4"],
            "{log:?}"
        );
    }
}
