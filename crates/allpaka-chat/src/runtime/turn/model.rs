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

    /// Call the model once, trying configured fallback routes on failover-eligible
    /// provider errors. Returns `(message, usage, answered_provider, answered_model)`.
    /// Unless `buffer_output` is set, text streams into message `index` and the
    /// session is saved at most every two seconds.
    pub(super) async fn call_model(
        &self,
        history: &[Message],
        index: usize,
        buffer_output: bool,
    ) -> Result<(Message, Value, String, String)> {
        let settings = self.settings;
        let system = &self.ctx.system;
        let step_system = if settings.mode == Mode::Goal {
            let state = self.s.lock().unwrap();
            format!("{system}\nCurrent goal origin: {}. Original user objective: {}.\nDurable milestone plan revision: {}. Use this exact base_revision in set_plan. Keep milestone IDs. Completed milestones must not be removed or repeated. Acceptance and evidence are reported claims, not independent proof. Current milestones: {}",serde_json::to_string(&state.goal)?,serde_json::to_string(&state.goal.as_ref().and_then(|goal|state.messages.get(goal.message_index)).map(|message|message.content.as_str()))?,state.plan_revision,serde_json::to_string(&state.plan)?)
        } else {
            system.clone()
        };
        let routes = std::iter::once((&self.ctx.provider, settings))
            .chain(self.ctx.fallbacks.iter().map(|(p, s)| (p, s)));
        let mut errors = Vec::new();
        for (attempt_idx, (route_provider, route_settings)) in routes.enumerate() {
            if attempt_idx > 0 {
                if let Err(err) = history
                    .iter()
                    .try_for_each(|m| context::validate_images(&m.images, &route_settings.provider))
                {
                    errors.push(format!(
                        "{} ({}): {err:#}",
                        route_settings.provider, route_settings.model
                    ));
                    continue;
                }
                // Drop any partial text/reasoning streamed by the failed primary
                // before the fallback route starts streaming into the same slot.
                let mut state = self.s.lock().unwrap();
                let message = &mut state.messages[index];
                message.content.clear();
                message.reasoning_content = None;
            }
            let copy = self.s.clone();
            let save_app = self.app.clone();
            let mut last_save = std::time::Instant::now();
            let mut model_span =
                self.trace
                    .span("model", &route_settings.model, Some(self.parent_span));
            model_span.set_provider(&route_settings.provider);
            let generated = provider::generate(
                &self.app.client,
                route_provider,
                route_settings,
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
            match generated {
                Ok((message, usage)) => {
                    return Ok((
                        message,
                        usage,
                        route_settings.provider.clone(),
                        route_settings.model.clone(),
                    ));
                }
                Err(err) => {
                    if self.ctx.fallbacks.is_empty() || !provider::is_failover_eligible(&err) {
                        return Err(err);
                    }
                    errors.push(format!(
                        "{} ({}): {err:#}",
                        route_settings.provider, route_settings.model
                    ));
                }
            }
        }
        anyhow::bail!("All configured providers failed: {}", errors.join("; "))
    }
}
