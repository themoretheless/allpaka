//! The guardrail stage of a turn: policies pinned once, the user input
//! checked before each model call that sees new input, and the reply
//! checked before it is kept.

use super::last_user_text;
use crate::*;

pub(crate) struct TurnGuardrails {
    prepared: Option<guardrail_policies::Prepared>,
    /// The last user text that passed the input check.
    guarded_input: Option<String>,
}

impl TurnGuardrails {
    /// Pin the selected policies into `settings` and check the latest user
    /// message once before anything else runs.
    pub(crate) fn prepare(
        app: &App,
        s: &SharedSession,
        settings: &mut Settings,
        trace: &observability::Trace,
        parent_span: usize,
    ) -> Result<Self> {
        let guardrails = settings
            .guardrails
            .as_ref()
            .map(|selection| selection.prepare(&guardrail_policies::Store::open(&app.data)?))
            .transpose()?;
        settings.prepared_guardrails = guardrails
            .as_ref()
            .map(|prepared| Arc::new(prepared.clone()));
        let mut guarded_input = None;
        if let Some(prepared) = &guardrails {
            let text = last_user_text(&s.lock().unwrap())
                .unwrap_or_default()
                .to_string();
            enforce_guardrail(prepared.input(&text)?, trace, parent_span)?;
            guarded_input = Some(text);
        }
        Ok(Self {
            prepared: guardrails,
            guarded_input,
        })
    }

    /// Check the latest user message of `history` unless it already passed.
    pub(crate) fn check_input(
        &mut self,
        history: &[Message],
        trace: &observability::Trace,
        parent_span: usize,
    ) -> Result<()> {
        if let Some(prepared) = &self.prepared {
            if let Some(user) = history.iter().rev().find(|message| message.role == "user") {
                if self.guarded_input.as_ref() != Some(&user.content) {
                    enforce_guardrail(prepared.input(&user.content)?, trace, parent_span)?;
                    self.guarded_input = Some(user.content.clone());
                }
            }
        }
        Ok(())
    }

    /// Whether the reply must be held back until its output check passes.
    pub(crate) fn buffers_output(&self) -> bool {
        self.prepared
            .as_ref()
            .is_some_and(|prepared| prepared.buffers_output())
    }

    /// Check a reply. The outer error is a check that could not run; the
    /// inner one is the enforcement result, after which the caller drops
    /// the reply.
    pub(crate) fn check_output(
        &self,
        text: &str,
        trace: &observability::Trace,
        parent_span: usize,
    ) -> Result<Result<()>> {
        match &self.prepared {
            Some(prepared) => Ok(enforce_guardrail(
                prepared.output(text)?,
                trace,
                parent_span,
            )),
            None => Ok(Ok(())),
        }
    }
}
