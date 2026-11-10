//! The persistence stage of a turn: every change a step makes to the
//! conversation is written to the session and saved straight away.

use super::Turn;
use crate::*;

impl Turn<'_> {
    /// Keep the reply in the placeholder message and save.
    pub(super) fn store_reply(&self, index: usize, message: Message, usage: Value) {
        {
            let mut s = self.s.lock().unwrap();
            s.messages[index] = message;
            s.usage = usage;
        }
        persist(self.app, self.s);
    }

    /// Drop a rejected reply, keeping its usage, and save.
    pub(super) fn discard_reply(&self, index: usize, usage: &Value) {
        {
            let mut state = self.s.lock().unwrap();
            state.messages[index] = Message::text("assistant", "");
            state.usage = usage.clone();
        }
        persist(self.app, self.s);
    }

    /// Append a message and save.
    pub(super) fn push_message(&self, message: Message) {
        self.s.lock().unwrap().messages.push(message);
        persist(self.app, self.s);
    }
}
