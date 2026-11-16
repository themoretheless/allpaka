//! Long generations taking turns, one step each.
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
use allpaka_model::Session;
use anyhow::anyhow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Condvar, Mutex as StdMutex};

/// What one decode step produced for a session.
pub(super) enum Next {
    /// The greedy token (the caller asked for greedy decoding).
    Token(u32),
    /// The full distribution.
    Logits(Vec<f32>),
}

/// A decode step waiting to be batched: the owner is blocked in
/// [`Gate::decode`] until its result is in, so its session is not touched by
/// anyone else meanwhile.
struct Pending {
    id: u64,
    token: u32,
    session: usize,
    greedy: bool,
}

#[derive(Default)]
struct State {
    /// Next ticket to hand out, and the ticket being served.
    next: u64,
    serving: u64,
    next_id: u64,
    pending: Vec<Pending>,
    done: HashMap<u64, std::result::Result<Next, String>>,
}

/// A fair turn: whoever asked first goes next. A party holds the turn
/// between its steps and gives it up at each step boundary (`pass`), or, at a
/// decode step, through [`Gate::decode`], which batches every party then
/// waiting to decode into one pass over the weights.
pub(super) struct Gate {
    state: StdMutex<State>,
    turn: Condvar,
}

impl Gate {
    /// A gate whose first turn (ticket 0) is already held by the caller.
    fn held() -> Self {
        Self {
            state: StdMutex::new(State {
                next: 1,
                ..State::default()
            }),
            turn: Condvar::new(),
        }
    }

    fn wait_turn<'a>(
        &self,
        mut state: std::sync::MutexGuard<'a, State>,
        ticket: u64,
    ) -> std::sync::MutexGuard<'a, State> {
        while state.serving != ticket {
            state = self.turn.wait(state).unwrap();
        }
        state
    }

    /// Wait for a turn.
    pub(super) fn enter(&self) {
        let mut state = self.state.lock().unwrap();
        let ticket = state.next;
        state.next += 1;
        drop(self.wait_turn(state, ticket));
    }

    /// End a turn.
    pub(super) fn leave(&self) {
        self.state.lock().unwrap().serving += 1;
        self.turn.notify_all();
    }

    /// End this turn and wait for the next one: anyone who asked meanwhile
    /// goes first.
    pub(super) fn pass(&self) {
        let mut state = self.state.lock().unwrap();
        state.serving += 1;
        let ticket = state.next;
        state.next += 1;
        self.turn.notify_all();
        drop(self.wait_turn(state, ticket));
    }

    /// One decode step of `session`, called holding the turn and returning
    /// holding it. The step is left with the gate and the turn passed on;
    /// whichever party holds the turn while the step is still unserved runs
    /// it together with every other waiting step (up to `MULTI_MAX`) as one
    /// batch. A party whose step someone else ran just waits for its turn.
    pub(super) fn decode(
        &self,
        model: &Model<'static>,
        token: u32,
        session: &mut Session,
        greedy: bool,
    ) -> Result<Next> {
        let mut state = self.state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.pending.push(Pending {
            id,
            token,
            session: session as *mut Session as usize,
            greedy,
        });
        state.serving += 1;
        let ticket = state.next;
        state.next += 1;
        self.turn.notify_all();
        let mut state = self.wait_turn(state, ticket);
        if let Some(done) = state.done.remove(&id) {
            return done.map_err(|e| anyhow!(e));
        }
        // Unserved at our turn: run every waiting step, ours first.
        let at = state
            .pending
            .iter()
            .position(|p| p.id == id)
            .expect("own step pending");
        let mut batch = vec![state.pending.remove(at)];
        while batch.len() < MULTI_MAX && !state.pending.is_empty() {
            batch.push(state.pending.remove(0));
        }
        drop(state);
        let results = run_batch(model, &batch);
        let mut state = self.state.lock().unwrap();
        let mut own = None;
        for (step, result) in batch.iter().zip(results) {
            if step.id == id {
                own = Some(result);
            } else {
                state.done.insert(step.id, result);
            }
        }
        drop(state);
        own.expect("own step in batch").map_err(|e| anyhow!(e))
    }
}

/// Most decode steps batched at once (llama's MMVQ takes up to 8 rows).
const MULTI_MAX: usize = 8;

fn run_batch(model: &Model<'static>, batch: &[Pending]) -> Vec<std::result::Result<Next, String>> {
    // SAFETY: each session's owner is blocked in `Gate::decode` until its
    // result is stored, and every step in a batch belongs to a different
    // owner (one pending step each), so these are the only references.
    let mut sessions: Vec<&mut Session> = batch
        .iter()
        .map(|p| unsafe { &mut *(p.session as *mut Session) })
        .collect();
    if let [step] = batch {
        let s = &mut *sessions[0];
        let one = if step.greedy {
            model.forward_greedy(step.token, s).map(Next::Token)
        } else {
            model.forward(step.token, s).map(Next::Logits)
        };
        return vec![one.map_err(|e| format!("{e:#}"))];
    }
    let tokens: Vec<u32> = batch.iter().map(|p| p.token).collect();
    match model.forward_multi(&tokens, &mut sessions) {
        Ok(all) => all
            .into_iter()
            .zip(batch)
            .map(|(logits, step)| {
                Ok(if step.greedy {
                    Next::Token(argmax(&logits))
                } else {
                    Next::Logits(logits)
                })
            })
            .collect(),
        Err(e) => batch.iter().map(|_| Err(format!("{e:#}"))).collect(),
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

thread_local! {
    /// The gate this thread's generation takes turns through, if any.
    static TURN: RefCell<Option<Arc<Gate>>> = const { RefCell::new(None) };
}

pub(super) fn set_turn(gate: Option<Arc<Gate>>) {
    TURN.with(|t| *t.borrow_mut() = gate);
}

pub(super) fn current() -> Option<Arc<Gate>> {
    TURN.with(|t| t.borrow().clone())
}

/// A decode step through this thread's gate (batched with the other
/// generations'); `None` when this thread takes turns with nobody.
pub(super) fn decode_step(
    model: &Model<'static>,
    token: u32,
    session: &mut Session,
    greedy: bool,
) -> Option<Result<Next>> {
    current().map(|gate| gate.decode(model, token, session, greedy))
}

/// Pass the turn at a step `decode_step` does not cover (a speculative round).
pub(super) fn pass_turn() {
    if let Some(gate) = current() {
        gate.pass();
    }
}

/// Generations taking turns at most, the dispatcher's own included: each one
/// more stretches every other's token gap by one step.
pub(super) const MAX_TURNS: usize = 4;

/// The gate shared with the running workers, and how many run.
static ACTIVE: StdMutex<Option<(Arc<Gate>, usize)>> = StdMutex::new(None);

pub(super) fn active() -> Option<Arc<Gate>> {
    ACTIVE
        .lock()
        .unwrap()
        .as_ref()
        .map(|(gate, _)| Arc::clone(gate))
}

fn workers() -> usize {
    ACTIVE.lock().unwrap().as_ref().map_or(0, |(_, n)| *n)
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
    pub(super) sessions: NamedSessions,
}

/// Whether a waiting request should run alongside the current one: a chat
/// completion for the same model, without a named session, while no worker
/// fewer than `MAX_TURNS` generations take turns. (One much shorter than the running request is served in between
/// instead; see `preemptible`.)
pub(super) fn eligible(requested: &str, shared: &Shared, connection: &QueuedConnection) -> bool {
    matches!(
        routes::classify(&connection.request_line).map(|m| m.route),
        Some(Route::ChatCompletions)
    ) && requested == shared.name
        && named_free(shared, connection)
        && workers() + 1 < MAX_TURNS
}

/// A request without `session_id`, or one whose named session is parked
/// (not in the running generation) or new; a malformed id is answered by
/// the ordinary path.
fn named_free(shared: &Shared, connection: &QueuedConnection) -> bool {
    match connection.request.get("session_id") {
        None => true,
        Some(id) => id
            .as_str()
            .filter(|id| !id.is_empty() && id.len() <= 128)
            .is_some_and(|id| {
                let parked = shared.sessions.lock().unwrap();
                parked.contains_key(id)
                    || (parked.len() < super::session::NAMED_SESSIONS_MAX
                        && !super::session::named_in_use(id))
            }),
    }
}

/// Start `connection` on a worker that takes turns with the caller, and return
/// the gate. A new gate's first turn is the caller's; on a gate already in
/// use the caller holds a turn by now. Either way the caller must `pass` after
/// each step and `leave` when its own generation ends.
pub(super) fn spawn(shared: Shared, connection: QueuedConnection) -> Arc<Gate> {
    let gate = {
        let mut active = ACTIVE.lock().unwrap();
        let (gate, workers) = active.get_or_insert_with(|| (Arc::new(Gate::held()), 0));
        *workers += 1;
        Arc::clone(gate)
    };
    let worker_gate = Arc::clone(&gate);
    let named = connection.request["session_id"].as_str().map(str::to_owned);
    std::thread::spawn(move || {
        worker_gate.enter();
        set_turn(Some(Arc::clone(&worker_gate)));
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
        // Decode steps take the turn through `decode_step` (batched).
        let pass = |_: &Model<'static>,
                    _: &Tokenizer,
                    _: &Template,
                    _: &memory::Budget,
                    _: usize,
                    in_decode: bool| {
            if !in_decode {
                worker_gate.pass();
            }
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
            named.as_ref().map(|id| (&shared.sessions, id.clone())),
        );
        finish(&worker_gate);
    });
    gate
}

/// The worker is done: the last one stops gating; then give up its turn.
fn finish(gate: &Arc<Gate>) {
    set_turn(None);
    let mut active = ACTIVE.lock().unwrap();
    if let Some((_, workers)) = active.as_mut().filter(|(g, _)| Arc::ptr_eq(g, gate)) {
        *workers -= 1;
        if *workers == 0 {
            *active = None;
        }
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
            for i in 0..200 {
                l.lock().unwrap().push(format!("w{i}"));
                std::thread::yield_now();
                g.pass();
            }
            g.leave();
        });
        // Let the worker queue up for its first turn.
        while gate.state.lock().unwrap().next < 2 {
            std::thread::yield_now();
        }
        for i in 0..200 {
            log.lock().unwrap().push(format!("m{i}"));
            std::thread::yield_now();
            gate.pass();
        }
        gate.leave();
        worker.join().unwrap();
        let log = log.lock().unwrap().clone();
        let expected: Vec<String> = (0..200)
            .flat_map(|i| [format!("m{i}"), format!("w{i}")])
            .collect();
        assert_eq!(log, expected, "{log:?}");
    }
}
