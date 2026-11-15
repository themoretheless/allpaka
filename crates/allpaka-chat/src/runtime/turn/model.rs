//! The provider-call stage of a turn step: the history the model sees and
//! one streamed generation into a placeholder assistant message.

use super::Turn;
use crate::*;

impl Turn<'_> {
    /// Fold pending steering into the conversation, record the step, and
    /// append the empty assistant message the reply streams into. Returns
    /// the history for the model and the index of that message.
    pub(super) fn begin_step(
        &self,
        step: usize,
        failure_loop: &mut progress::FailureLoop,
    ) -> (Vec<Message>, usize) {
        let mut s = self.s.lock().unwrap();
        let steers = std::mem::take(&mut s.steering);
        if !steers.is_empty() {
            failure_loop.reset();
        }
        for text in steers {
            s.messages.push(Message::text(
                "user",
                format!("User steering for the current task: {text}"),
            ));
        }
        s.step = step;
        let history = compact::history(&s);
        let index = s.messages.len();
        s.messages.push(Message::text("assistant", ""));
        (history, index)
    }

    /// Call the model once. Unless `buffer_output` is set, text streams into
    /// message `index` and the session is saved at most every two seconds.
    pub(super) async fn call_model(
        &self,
        history: &[Message],
        index: usize,
        buffer_output: bool,
    ) -> Result<(Message, Value)> {
        let settings = self.settings;
        let copy = self.s.clone();
        let save_app = self.app.clone();
        let mut last_save = std::time::Instant::now();
        let mut model_span = self
            .trace
            .span("model", &settings.model, Some(self.parent_span));
        model_span.set_provider(&settings.provider);
        let system = &self.ctx.system;
        let step_system = if settings.mode == Mode::Goal {
            let state = self.s.lock().unwrap();
            format!("{system}\nCurrent goal origin: {}. Original user objective: {}.\nDurable milestone plan revision: {}. Use this exact base_revision in set_plan. Keep milestone IDs. Completed milestones must not be removed or repeated. Acceptance and evidence are reported claims, not independent proof. Current milestones: {}",serde_json::to_string(&state.goal)?,serde_json::to_string(&state.goal.as_ref().and_then(|goal|state.messages.get(goal.message_index)).map(|message|message.content.as_str()))?,state.plan_revision,serde_json::to_string(&state.plan)?)
        } else {
            system.clone()
        };
        let generated = provider::generate(
            &self.app.client,
            &self.ctx.provider,
            settings,
            &step_system,
            history,
            &self.ctx.schemas,
            move |text, reasoning| {
                if buffer_output {
                    return;
                }
                {
                    let mut state = copy.lock().unwrap();
                    let message = &mut state.messages[index];
                    message.content.push_str(&text);
                    if !reasoning.is_empty() {
                        message
                            .reasoning_content
                            .get_or_insert_with(String::new)
                            .push_str(&reasoning);
                    }
                }
                if last_save.elapsed().as_secs() >= 2 {
                    persist(&save_app, &copy);
                    last_save = std::time::Instant::now();
                }
            },
        )
        .await;
        model_span.finish(
            if generated.as_ref().is_ok_and(|(m, _)| !m.truncated) {
                "completed"
            } else if generated.is_ok() {
                "token_limit"
            } else {
                "failed"
            },
            generated
                .as_ref()
                .map(|(_, usage)| usage)
                .unwrap_or(&Value::Null),
        );
        generated
    }
}
