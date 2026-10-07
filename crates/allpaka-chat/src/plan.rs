//! Durable milestone identities and reported acceptance evidence on the existing plan.
use crate::types::{Mode, PlanCheckpoint, PlanItem, Session};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
const MAX_HISTORY_BYTES: usize = 1024 * 1024;
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 80
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
pub(crate) fn validate_steps(steps: &[PlanItem]) -> Result<()> {
    if steps.len() > 30 {
        bail!("Select at most 30 plan steps");
    }
    if serde_json::to_vec(steps)?.len() > 64 * 1024 {
        bail!("Plan exceeds 64 KiB");
    }
    let mut ids = HashSet::new();
    for step in steps {
        if step.title.trim().is_empty()
            || step.title.len() > 500
            || !matches!(
                step.status.as_str(),
                "pending" | "in_progress" | "completed"
            )
            || !step.id.is_empty() && (!valid_id(&step.id) || !ids.insert(&step.id))
            || step.acceptance.len() > 4
            || step.evidence.len() > 4
            || step
                .acceptance
                .iter()
                .any(|v| v.trim().is_empty() || v.len() > 1000)
            || step
                .evidence
                .iter()
                .any(|v| v.trim().is_empty() || v.len() > 2000)
        {
            bail!("Invalid bounded plan step");
        }
    }
    Ok(())
}
fn checkpoint(session: &mut Session, source: &str) -> Result<()> {
    session.plan_revision = session
        .plan_revision
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("Plan revision exhausted"))?;
    let recorded_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64;
    session.plan_checkpoints.push(PlanCheckpoint {
        goal_origin: session.goal.clone(),
        goal_id: session.goal.as_ref().map(|goal| goal.id.clone()),
        revision: session.plan_revision,
        recorded_ms,
        source: source.into(),
        steps: session.plan.clone(),
    });
    while session.plan_checkpoints.len() > 30
        || serde_json::to_vec(&session.plan_checkpoints)?.len() > MAX_HISTORY_BYTES
    {
        if session.plan_checkpoints.len() == 1 {
            bail!("Plan checkpoint exceeds history bound");
        }
        session.plan_checkpoints.remove(0);
    }
    Ok(())
}
/// A new user goal gets its own plan; prior completed work remains in checkpoints.
pub(crate) fn start_goal(session: &mut Session, message_index: usize) -> Result<()> {
    if !session
        .messages
        .get(message_index)
        .is_some_and(|message| message.role == "user")
    {
        bail!("Goal origin must refer to a user message");
    }
    let mut next = session.clone();
    next.goal = Some(crate::types::GoalOrigin {
        id: crate::evaluation::new_id(),
        message_index,
    });
    if !next.plan.is_empty() {
        next.plan.clear();
        checkpoint(&mut next, "user")?;
    }
    *session = next;
    Ok(())
}
pub(crate) fn restore(session: &mut Session) -> Result<()> {
    validate_steps(&session.plan)?;
    if let Some(goal) = &session.goal {
        if goal.id.is_empty()
            || goal.id.len() > 80
            || !goal
                .id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
            || !session
                .messages
                .get(goal.message_index)
                .is_some_and(|message| message.role == "user")
        {
            bail!("Invalid persisted goal origin");
        }
    }
    if session.plan_checkpoints.len() > 30
        || serde_json::to_vec(&session.plan_checkpoints)?.len() > MAX_HISTORY_BYTES
    {
        bail!("Invalid plan history bounds");
    }
    let mut origins = HashMap::new();
    if let Some(goal) = &session.goal {
        origins.insert(goal.id.as_str(), goal.message_index);
    }
    let mut previous = 0;
    for entry in &session.plan_checkpoints {
        validate_steps(&entry.steps)?;
        if entry.revision <= previous
            || entry.revision > session.plan_revision
            || !matches!(entry.source.as_str(), "user" | "agent" | "legacy_recovery")
            || entry.steps.iter().any(|step| step.id.is_empty())
            || entry.goal_id.as_ref().is_some_and(|id| {
                id.is_empty()
                    || id.len() > 80
                    || !id
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
            })
        {
            bail!("Invalid plan checkpoint");
        }
        if let Some(origin) = &entry.goal_origin {
            if entry.goal_id.as_deref() != Some(origin.id.as_str())
                || !session
                    .messages
                    .get(origin.message_index)
                    .is_some_and(|message| message.role == "user")
            {
                bail!("Invalid checkpoint goal origin");
            }
            if origins
                .insert(origin.id.as_str(), origin.message_index)
                .is_some_and(|index| index != origin.message_index)
            {
                bail!("Conflicting checkpoint goal origins");
            }
        }
        previous = entry.revision;
    }
    if let Some(last) = session.plan_checkpoints.last() {
        if last.revision != session.plan_revision
            || serde_json::to_value(&last.steps)? != serde_json::to_value(&session.plan)?
            || !session.plan.is_empty()
                && last.goal_id.as_deref() != session.goal.as_ref().map(|goal| goal.id.as_str())
        {
            bail!("Plan differs from its checkpoint");
        }
    } else if session.plan_revision != 0 {
        bail!("Missing plan checkpoint");
    }
    if session.plan.iter().any(|step| step.id.is_empty()) {
        for step in &mut session.plan {
            if step.id.is_empty() {
                step.id = crate::evaluation::new_id();
            }
        }
        checkpoint(session, "legacy_recovery")?;
    }
    Ok(())
}
/// Readiness is based on reported milestone data, never independent proof.
pub(crate) fn reported_ready(session: &Session) -> bool {
    !session.plan.is_empty()
        && session.plan.iter().all(|step| {
            step.status == "completed" && !step.acceptance.is_empty() && !step.evidence.is_empty()
        })
}

pub(crate) fn update(
    session: &mut Session,
    mut steps: Vec<PlanItem>,
    base: Option<u64>,
    source: &str,
    allow_reopen: bool,
) -> Result<()> {
    validate_steps(&steps)?;
    if !matches!(source, "user" | "agent") || allow_reopen && source != "user" {
        bail!("Invalid plan update authority");
    }
    if base.is_some_and(|revision| revision != session.plan_revision) {
        bail!("Plan revision conflict; reload the current plan");
    }
    let goal = session.settings.mode == Mode::Goal;
    if goal && (base.is_none() || steps.is_empty()) {
        bail!("Goal plans require a base_revision and at least one milestone");
    }
    for step in &mut steps {
        if step.id.is_empty() {
            let matching = session
                .plan
                .iter()
                .filter(|old| old.title == step.title)
                .collect::<Vec<_>>();
            if matching.len() > 1 {
                bail!("Use milestone IDs for duplicate titles");
            }
            step.id = matching
                .first()
                .map_or_else(crate::evaluation::new_id, |old| old.id.clone());
        }
        let old = session.plan.iter().find(|old| old.id == step.id);
        let unchanged = old.is_some_and(|old| {
            old.title == step.title
                && old.status == step.status
                && old.acceptance == step.acceptance
                && old.evidence == step.evidence
        });
        let explicit_reopen = allow_reopen
            && old.is_some_and(|old| old.status == "completed")
            && step.status == "pending";
        if goal
            && !unchanged
            && !explicit_reopen
            && (step.acceptance.is_empty()
                || step.status == "completed" && step.evidence.is_empty())
        {
            bail!("Goal milestones require acceptance criteria; completion requires reported evidence");
        }
    }
    validate_steps(&steps)?;
    if goal && !allow_reopen {
        for old in &session.plan {
            if old.status == "completed" {
                let next = steps.iter().find(|step| step.id == old.id).ok_or_else(|| {
                    anyhow::anyhow!("Completed Goal milestones cannot be silently removed")
                })?;
                if next.status != "completed"
                    || next.title != old.title
                    || next.acceptance != old.acceptance
                    || next.evidence != old.evidence
                {
                    bail!("Completed Goal milestones require an explicit user reopen to change");
                }
            }
        }
    }
    session.plan = steps;
    checkpoint(session, source)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn step(title: &str, status: &str) -> PlanItem {
        PlanItem {
            id: String::new(),
            title: title.into(),
            status: status.into(),
            acceptance: vec!["Acceptance criterion".into()],
            evidence: if status == "completed" {
                vec!["Reported check result".into()]
            } else {
                vec![]
            },
        }
    }
    #[test]
    fn checkpoint_cannot_substitute_another_goal() {
        let settings = serde_json::from_value(serde_json::json!({"project_id":"default","provider":"local","model":"mock","mode":"goal"})).unwrap();
        let mut session = Session::new("session".into(), settings);
        session
            .messages
            .push(crate::types::Message::text("user", "Current objective"));
        start_goal(&mut session, 0).unwrap();
        update(
            &mut session,
            vec![step("Done", "completed")],
            Some(0),
            "agent",
            false,
        )
        .unwrap();
        let mut substituted = session.clone();
        substituted.plan_checkpoints.last_mut().unwrap().goal_id = Some("other-goal".into());
        assert!(restore(&mut substituted).is_err());
        let mut malformed = session.clone();
        malformed.plan_checkpoints.last_mut().unwrap().goal_id = Some("invalid/id".into());
        assert!(restore(&mut malformed).is_err());
        session.goal.as_mut().unwrap().message_index = 99;
        assert!(restore(&mut session).is_err());
    }
    #[test]
    fn new_goal_archives_old_plan_without_inheriting_completion() {
        let settings = serde_json::from_value(serde_json::json!({"project_id":"default","provider":"local","model":"mock","mode":"goal"})).unwrap();
        let mut session = Session::new("session".into(), settings);
        session
            .messages
            .push(crate::types::Message::text("user", "First objective"));
        start_goal(&mut session, 0).unwrap();
        let first = session.goal.as_ref().unwrap().id.clone();
        update(
            &mut session,
            vec![step("Done", "completed")],
            Some(0),
            "agent",
            false,
        )
        .unwrap();
        assert!(reported_ready(&session));
        session
            .messages
            .push(crate::types::Message::text("user", "Second objective"));
        start_goal(&mut session, 1).unwrap();
        assert_ne!(session.goal.as_ref().unwrap().id, first);
        assert!(!reported_ready(&session));
        assert!(session.plan.is_empty());
        assert_eq!(
            session.plan_checkpoints[0].goal_id.as_deref(),
            Some(first.as_str())
        );
        assert_eq!(session.plan_checkpoints[0].steps[0].status, "completed");
        assert_eq!(
            session.plan_checkpoints[0]
                .goal_origin
                .as_ref()
                .unwrap()
                .message_index,
            0
        );
        let mut invalid_origin = session.clone();
        invalid_origin.plan_checkpoints[0]
            .goal_origin
            .as_mut()
            .unwrap()
            .message_index = 99;
        assert!(restore(&mut invalid_origin).is_err());
        let mut conflicting_origin = session.clone();
        conflicting_origin
            .plan_checkpoints
            .last_mut()
            .unwrap()
            .goal_origin
            .as_mut()
            .unwrap()
            .message_index = 0;
        assert!(restore(&mut conflicting_origin).is_err());
        let mut restored: Session =
            serde_json::from_value(serde_json::to_value(&session).unwrap()).unwrap();
        restore(&mut restored).unwrap();
        assert_eq!(restored.goal.unwrap().message_index, 1);
        assert!(start_goal(&mut session, 99).is_err());
    }
    #[test]
    fn reported_readiness_requires_every_milestone_with_criteria_and_evidence() {
        let settings = serde_json::from_value(serde_json::json!({"project_id":"default","provider":"local","model":"mock","mode":"goal"})).unwrap();
        let mut session = Session::new("session".into(), settings);
        assert!(!reported_ready(&session));
        session.plan = vec![step("done", "completed"), step("remaining", "pending")];
        assert!(!reported_ready(&session));
        session.plan[1] = step("remaining", "completed");
        assert!(reported_ready(&session));
        session.plan[0].evidence.clear();
        assert!(!reported_ready(&session));
        session.plan[0].evidence.push("Reported evidence".into());
        session.plan[0].acceptance.clear();
        assert!(!reported_ready(&session));
    }
    #[test]
    fn durable_ids_checkpoint_conflicts_and_completed_preservation() {
        let mut settings = serde_json::from_value::<crate::types::Settings>(serde_json::json!({"project_id":"default","provider":"local","model":"mock","mode":"auto"})).unwrap();
        settings.mode = Mode::Goal;
        let mut session = Session::new("session".into(), settings);
        update(
            &mut session,
            vec![step("first", "completed"), step("second", "pending")],
            Some(0),
            "agent",
            false,
        )
        .unwrap();
        let first = session.plan[0].id.clone();
        let mut restored: Session =
            serde_json::from_slice(&serde_json::to_vec(&session).unwrap()).unwrap();
        restore(&mut restored).unwrap();
        assert_eq!(restored.plan[0].id, first);
        let old = serde_json::to_value(&restored).unwrap();
        let mut stale = restored.clone();
        assert!(update(&mut stale, session.plan.clone(), Some(0), "agent", false).is_err());
        assert_eq!(serde_json::to_value(stale).unwrap(), old);
        let mut removed = restored.clone();
        assert!(update(
            &mut removed,
            vec![step("second", "pending")],
            Some(1),
            "agent",
            false
        )
        .is_err());
        let mut incomplete = step("third", "completed");
        incomplete.evidence.clear();
        assert!(update(&mut restored, vec![incomplete], Some(1), "agent", false).is_err());
        let mut next = session.plan.clone();
        next[0].status = "pending".into();
        update(&mut session, next, Some(1), "user", true).unwrap();
        assert_eq!(session.plan_revision, 2);
        assert_eq!(session.plan_checkpoints[0].steps[0].status, "completed");
    }
    #[test]
    fn legacy_goal_can_be_filled_incrementally_without_recertifying_old_completion() {
        let settings=serde_json::from_value(serde_json::json!({"project_id":"default","provider":"local","model":"mock","mode":"goal"})).unwrap();
        let mut session = Session::new("session".into(), settings);
        let mut done = step("old done", "completed");
        done.acceptance.clear();
        done.evidence.clear();
        let mut pending = step("old pending", "pending");
        pending.acceptance.clear();
        session.plan = vec![done, pending];
        restore(&mut session).unwrap();
        let mut next = session.plan.clone();
        next[1].acceptance = vec!["New criterion".into()];
        update(&mut session, next, Some(1), "user", false).unwrap();
        assert!(session.plan[0].acceptance.is_empty());
        let mut reopened = session.plan.clone();
        reopened[0].status = "pending".into();
        update(&mut session, reopened, Some(2), "user", true).unwrap();
        let mut invalid = session.plan.clone();
        invalid[0].status = "completed".into();
        assert!(update(&mut session, invalid, Some(3), "agent", false).is_err());
    }
    #[test]
    fn legacy_recovery_and_checkpoint_tamper_rejection() {
        let mut session = Session::new("session".into(), serde_json::from_value::<crate::types::Settings>(serde_json::json!({"project_id":"default","provider":"local","model":"mock","mode":"auto"})).unwrap());
        session.plan = vec![step("legacy", "pending")];
        restore(&mut session).unwrap();
        assert!(!session.plan[0].id.is_empty());
        assert_eq!(session.plan_revision, 1);
        session.plan[0].status = "completed".into();
        assert!(restore(&mut session).is_err());
    }
}
