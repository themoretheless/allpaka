//! Watch a running `serve` as a phase plus conditions.
//!
//! The shape is borrowed from an orchestrator's status model: `phase` is a
//! one-word summary, conditions carry the detail, and the useful thing to look
//! at is the transition between the two. Everything reported here is measured
//! from the four read-only endpoints `serve` already has - nothing is inferred
//! from a configured expectation.

use crate::client;
use anyhow::Result;
use serde_json::Value;
use std::io::ErrorKind;
use std::time::{Duration, Instant};

/// The outcome of one endpoint call, timed. A probe that times out is not a
/// failed probe: `/stats` is answered after the model lock, so a long
/// generation holds it and that is the server working, not broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Probe {
    Ok(Duration),
    TimedOut(Duration),
    /// Nothing came back, or nothing was asked yet.
    #[default]
    Failed,
}

impl Probe {
    fn waited(&self) -> Option<Duration> {
        match self {
            Self::Ok(d) | Self::TimedOut(d) => Some(*d),
            Self::Failed => None,
        }
    }

    fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }
}

/// What one poll round measured. The two context numbers are deliberately
/// separate: `/stats` reports the live session, `/resources` the value as of
/// the end of the last completed generation, which makes the second one a
/// progress counter and the first one useless for that.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Facts {
    pub health: Probe,
    pub models: Probe,
    pub stats: Probe,
    pub resources: Probe,
    pub model_count: Option<usize>,
    /// The id `/stats` reports. `/v1/models` is dispatched after the model lock
    /// like `/stats` is, so during a generation this is the only proof on hand
    /// that a model is actually being served.
    pub stats_model: Option<String>,
    pub context_used: Option<u64>,
    pub context_capacity: Option<u64>,
    pub cache_entries: Option<u64>,
    pub cache_bytes: Option<u64>,
    pub limit_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    pub last_generation_context: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Stopped,
    Starting,
    Ready,
    Generating,
    Degraded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    pub name: &'static str,
    pub held: bool,
    /// Why the condition holds or does not, in measured terms. Empty when the
    /// name says it all.
    pub reason: String,
}

impl Condition {
    fn because(name: &'static str, held: bool, reason: impl Into<String>) -> Self {
        Self { name, held, reason: reason.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    pub phase: Phase,
    pub conditions: Vec<Condition>,
}

/// A generation is in flight when `/stats` waits on the model lock while
/// `/resources` - served before the queue, so never blocked by it - does not.
/// The comparison is between two probes of the same round-trip cost on the
/// same connection style, which is why it needs no per-machine constant; the
/// factor only keeps one slow poll from flipping the phase.
const LOCK_CONTENTION_FACTOR: u32 = 4;

/// `serve` reports an unbudgeted memory limit as `u64::MAX`, not as zero.
const UNLIMITED: u64 = u64::MAX / 2;
const LOCK_CONTENTION_FLOOR: Duration = Duration::from_millis(25);

pub fn classify(facts: &Facts, previous: Option<&Facts>) -> Status {
    let reachable = facts.health.is_ok();
    let named_by_stats = facts.stats_model.is_some();
    let admitted = facts
        .model_count
        .unwrap_or(if named_by_stats { 1 } else { 0 });
    let stats_ok = facts.stats.is_ok();
    let resources_ok = facts.resources.is_ok();

    let busy_reason = busy_reason(facts, previous);
    // Only the two probes that answer before the model lock (`/health`,
    // `/resources`) can tell an outage from a busy engine. A post-lock probe
    // that goes unanswered during a generation is the generation, so the
    // conditions built on it stay held and say so.
    let lock_held = reachable && busy_reason.is_some();
    let blocked_by_lock = |probe: Probe| lock_held && matches!(probe, Probe::TimedOut(_));

    let phase = if !reachable {
        Phase::Stopped
    } else if admitted == 0 && facts.model_count.is_some() {
        Phase::Starting
    } else if busy_reason.is_some() {
        Phase::Generating
    } else if !stats_ok || !resources_ok || admitted == 0 {
        Phase::Degraded
    } else {
        Phase::Ready
    };

    let mut conditions = vec![Condition::because(
        "Reachable",
        reachable,
        match facts.health {
            Probe::Ok(d) => format!("{}ms", ms(d)),
            Probe::TimedOut(d) => format!("no answer in {}ms", ms(d)),
            Probe::Failed => "connection refused or unreadable".into(),
        },
    )];

    conditions.push(Condition::because(
        "ModelsAdmitted",
        admitted > 0 || blocked_by_lock(facts.models),
        match (facts.model_count, named_by_stats) {
            (Some(n), _) => format!("{n} endpoint(s)"),
            (None, true) => "/v1/models was behind the lock; /stats named one".into(),
            (None, false) if blocked_by_lock(facts.models) => {
                "/v1/models held by the model lock".into()
            }
            (None, false) => "/v1/models did not answer".into(),
        },
    ));

    conditions.push(Condition::because(
        "EngineResponsive",
        stats_ok || blocked_by_lock(facts.stats),
        if blocked_by_lock(facts.stats) && !stats_ok {
            "held by the model lock".into()
        } else {
            probe_reason(facts.stats, "queues behind the model lock")
        },
    ));

    conditions.push(Condition::because(
        "ResourcesSampled",
        resources_ok,
        probe_reason(facts.resources, "served before the queue"),
    ));

    conditions.push(Condition::because(
        "Generating",
        busy_reason.is_some(),
        busy_reason.unwrap_or_default(),
    ));

    // These two conditions are read out of the `/stats` body, so a probe held
    // by the model lock leaves them unknown rather than false. Calling unknown
    // "gone" would make every generation look like the cache was dropped.
    let unobserved = if stats_ok || !lock_held {
        None
    } else {
        Some("not observed (model lock held)".to_string())
    };
    let headroom = match (facts.limit_bytes, facts.reserved_bytes) {
        (Some(limit), Some(reserved)) if limit >= UNLIMITED => (
            true,
            format!("no cap set, {:.1} GiB reserved", gib(reserved)),
        ),
        (Some(limit), Some(reserved)) => {
            let ratio = reserved as f64 / limit as f64;
            (
                reserved < limit,
                format!(
                    "{:.1}/{:.1} GiB reserved ({:.0}%)",
                    gib(reserved),
                    gib(limit),
                    ratio * 100.0
                ),
            )
        }
        _ => (true, "not reported".into()),
    };
    let (held, reason) = match &unobserved {
        Some(reason) => (true, reason.clone()),
        None => headroom,
    };
    conditions.push(Condition::because("MemoryHeadroom", held, reason));

    let cache = match (facts.cache_entries, facts.cache_bytes) {
        // Residency, not reuse: `/stats` reports what is held, so this says
        // nothing about hit rate until the counters exist.
        (Some(entries), Some(bytes)) => (
            entries > 0,
            format!(
                "{entries} entries, {:.1} MiB held (residency, not hits)",
                bytes as f64 / (1u64 << 20) as f64
            ),
        ),
        _ => (true, "not reported".into()),
    };
    let (held, reason) = match &unobserved {
        Some(reason) => (true, reason.clone()),
        None => cache,
    };
    conditions.push(Condition::because(
        "PrefixCacheResident", held, reason,
    ));

    Status { phase, conditions }
}

fn busy_reason(facts: &Facts, previous: Option<&Facts>) -> Option<String> {
    let elapsed = |probe: Probe| probe.waited().unwrap_or(Duration::ZERO);
    if matches!(facts.stats, Probe::TimedOut(_)) {
        return Some(format!(
            "model lock held {}ms without answering",
            ms(elapsed(facts.stats))
        ));
    }
    let stats = elapsed(facts.stats);
    let resources = elapsed(facts.resources);
    if facts.stats.is_ok()
        && facts.resources.is_ok()
        && stats >= LOCK_CONTENTION_FLOOR
        && stats > resources.saturating_mul(LOCK_CONTENTION_FACTOR)
    {
        return Some(format!(
            "stats waited {}ms against resources {}ms",
            ms(stats),
            ms(resources)
        ));
    }
    // A completed generation moves the context captured at its end. This fires
    // once per generation rather than once per token, which is all a poll at
    // this cadence could resolve anyway.
    let progressed = previous
        .and_then(|p| Some((p.last_generation_context?, facts.last_generation_context?)))
        .filter(|(before, now)| now > before)
        .map(|(before, now)| format!("context went {} -> {} since last poll", before, now));
    progressed
}

fn probe_reason(probe: Probe, note: &str) -> String {
    match probe {
        Probe::Ok(d) => format!("{}ms, {note}", ms(d)),
        Probe::TimedOut(d) => format!("no answer in {}ms", ms(d)),
        Probe::Failed => format!("call failed, {note}"),
    }
}

fn ms(d: Duration) -> u64 {
    d.as_millis().try_into().unwrap_or(u64::MAX)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Lines describing what changed between two statuses. Empty means the phase
/// and every condition are as they were.
pub fn diff(previous: &Status, current: &Status) -> Vec<String> {
    let mut lines = Vec::new();
    if previous.phase != current.phase {
        lines.push(format!("{:?} -> {:?}", previous.phase, current.phase));
    }
    for condition in &current.conditions {
        let before = previous
            .conditions
            .iter()
            .find(|c| c.name == condition.name);
        let turned = before.is_none_or(|b| b.held != condition.held);
        if turned {
            let sign = if condition.held { '+' } else { '-' };
            lines.push(format!(
                "{sign}{}{}",
                condition.name,
                if condition.reason.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", condition.reason)
                }
            ));
        } else if condition.reason != before.map(|b| b.reason.as_str()).unwrap_or("") {
            lines.push(format!("~{} ({})", condition.name, condition.reason));
        }
    }
    lines
}

/// Whether a set of changes deserves a line. A `~` entry is the same condition
/// holding the same way with a different measurement attached, and every
/// measurement already appears on the line of the last real transition, so
/// reprinting them would turn a watch into a latency log.
pub fn is_transition(changes: &[String]) -> bool {
    changes.iter().any(|c| !c.starts_with('~'))
}

/// One line of the measurements behind the current status, so a reader can
/// disagree with a threshold using numbers rather than by reading the source.
pub fn measurements(facts: &Facts) -> String {
    let probe = |p: Probe| match p {
        Probe::Ok(d) => format!("{}ms", ms(d)),
        Probe::TimedOut(d) => format!("timeout {}ms", ms(d)),
        Probe::Failed => "failed".into(),
    };
    format!(
        "health={} models={} stats={} resources={} context={}/{} mem={}",
        probe(facts.health),
        probe(facts.models),
        probe(facts.stats),
        probe(facts.resources),
        facts
            .context_used
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into()),
        facts
            .context_capacity
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into()),
        match (facts.reserved_bytes, facts.limit_bytes) {
            (Some(r), Some(l)) if l >= UNLIMITED => format!("{:.1} GiB/uncapped", gib(r)),
            (Some(r), Some(l)) if l > 0 => format!("{:.1}/{:.1} GiB", gib(r), gib(l)),
            _ => "-".into(),
        },
    )
}

fn is_timeout(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|io| matches!(io.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut))
}

fn probe_endpoint(addr: &str, path: &str, budget: Duration) -> (Probe, Option<Value>) {
    let started = Instant::now();
    match client::get_within(addr, path, budget) {
        Ok(value) => (Probe::Ok(started.elapsed()), Some(value)),
        Err(error) if is_timeout(&error) => (Probe::TimedOut(started.elapsed()), None),
        Err(_) => (Probe::Failed, None),
    }
}

fn number(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn poll(addr: &str, budget: Duration) -> Facts {
    let mut facts = Facts::default();
    let (health, _) = probe_endpoint(addr, "/health", budget);
    facts.health = health;

    let (models, body) = probe_endpoint(addr, "/v1/models", budget);
    facts.models = models;
    facts.model_count = body
        .as_ref()
        .and_then(|v| v.get("data").and_then(Value::as_array))
        .map(Vec::len);

    let (stats, body) = probe_endpoint(addr, "/stats", budget);
    facts.stats = stats;
    if let Some(stats) = body.as_ref() {
        facts.context_used = number(stats, "context_used");
        facts.context_capacity = number(stats, "context_capacity");
        facts.cache_entries = number(stats, "prefix_cache_entries");
        facts.cache_bytes = number(stats, "prefix_cache_bytes");
        facts.stats_model = stats["model"].as_str().map(str::to_owned);
        facts.limit_bytes = number(&stats["memory_admission"], "limit_bytes");
        facts.reserved_bytes = number(&stats["memory_admission"], "reserved_bytes");
    }

    let (resources, body) = probe_endpoint(addr, "/resources", budget);
    facts.resources = resources;
    facts.last_generation_context = body
        .as_ref()
        .and_then(|v| number(v, "context_used"));

    facts
}

fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds_of_day = now.as_secs() % 86_400;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60,
        now.subsec_millis()
    )
}

fn status_json(facts: &Facts, status: &Status, changes: &[String]) -> String {
    let json = serde_json::json!({
        "ts": timestamp(),
        "phase": format!("{:?}", status.phase),
        "conditions": status.conditions.iter().map(|c| serde_json::json!({
            "name": c.name, "held": c.held, "reason": c.reason,
        })).collect::<Vec<_>>(),
        "changed": changes,
        "measurements": measurements(facts),
    });
    // Serialization of a hand-built value cannot fail; a `let-else` here would
    // only hide a bug in the shape above.
    serde_json::to_string(&json).unwrap_or_else(|_| "{}".into())
}

/// Poll `serve` and print only what moved, until the process is interrupted.
pub fn watch(addr: &str, interval: Duration, json: bool) -> Result<()> {
    let budget = interval.max(Duration::from_millis(50));
    let mut previous_facts: Option<Facts> = None;
    let mut previous: Option<Status> = None;
    loop {
        let tick = Instant::now();
        let facts = poll(addr, budget);
        let status = classify(&facts, previous_facts.as_ref());
        let changes = previous
            .as_ref()
            .map(|p| diff(p, &status))
            .unwrap_or_default();
        let moved = previous.is_none() || is_transition(&changes);
        if moved {
            if json {
                println!("{}", status_json(&facts, &status, &changes));
            } else {
                // `~` lines are the same condition holding the same way with a
                // different measurement attached; the measurements column already
                // carries them, so repeating them per transition is just noise.
                // They stay in the JSON, where a script may want the drift.
                let joined = changes
                    .iter()
                    .filter(|c| !c.starts_with('~'))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("  ");
                let joined = if joined.is_empty() {
                    String::new()
                } else {
                    format!("{joined}  ")
                };
                println!(
                    "{}  phase={:?}  {joined}{}",
                    timestamp(),
                    status.phase,
                    measurements(&facts)
                );
            }
        }
        previous = Some(status);
        previous_facts = Some(facts);
        std::thread::sleep(interval.saturating_sub(tick.elapsed()));
    }
}

#[cfg(test)]
mod tests {
    use super::{classify, diff, is_transition, measurements, Facts, Phase, Probe};
    use std::time::Duration;

    fn ok(elapsed_ms: u64) -> Probe {
        Probe::Ok(Duration::from_millis(elapsed_ms))
    }

    fn served() -> Facts {
        Facts {
            health: ok(1),
            models: ok(1),
            stats: ok(2),
            resources: ok(1),
            model_count: Some(1),
            context_used: Some(0),
            context_capacity: Some(8192),
            cache_entries: Some(0),
            cache_bytes: Some(0),
            limit_bytes: Some(96 << 30),
            reserved_bytes: Some(40 << 30),
            stats_model: Some("qwen3-0.6b-Q8_0".into()),
            last_generation_context: Some(0),
        }
    }

    #[test]
    fn nothing_answering_is_stopped_not_degraded() {
        let facts = Facts {
            health: Probe::Failed,
            ..Default::default()
        };
        assert_eq!(classify(&facts, None).phase, Phase::Stopped);
    }

    #[test]
    fn live_server_with_no_model_is_starting() {
        let facts = Facts {
            model_count: Some(0),
            stats_model: None,
            ..served()
        };
        assert_eq!(classify(&facts, None).phase, Phase::Starting);
    }

    #[test]
    fn lock_contention_reads_as_generating() {
        let facts = Facts {
            stats: ok(400),
            ..served()
        };
        assert_eq!(classify(&facts, None).phase, Phase::Generating);
    }

    #[test]
    fn a_timeout_is_generating_not_failed() {
        let facts = Facts {
            stats: Probe::TimedOut(Duration::from_secs(25)),
            ..served()
        };
        let status = classify(&facts, None);
        assert_eq!(status.phase, Phase::Generating);
        assert!(status
            .conditions
            .iter()
            .any(|c| c.name == "EngineResponsive" && c.held));
    }

    #[test]
    fn a_slow_probe_alone_does_not_look_like_contention() {
        // Both endpoints slow together is a loaded machine or a remote address,
        // not the model lock, so the ratio matters more than the absolute.
        let facts = Facts {
            stats: ok(400),
            resources: ok(380),
            ..served()
        };
        assert_eq!(classify(&facts, None).phase, Phase::Ready);
    }

    #[test]
    fn context_advancing_between_polls_counts_as_a_generation() {
        let before = served();
        let after = Facts {
            last_generation_context: Some(137),
            context_used: Some(137),
            ..served()
        };
        assert_eq!(classify(&after, Some(&before)).phase, Phase::Generating);
        assert_eq!(classify(&after, Some(&after)).phase, Phase::Ready);
    }

    #[test]
    fn a_served_endpoint_that_stops_answering_is_degraded() {
        let facts = Facts {
            resources: Probe::Failed,
            ..served()
        };
        assert_eq!(classify(&facts, None).phase, Phase::Degraded);
    }

    #[test]
    fn residency_is_reported_as_residency_not_as_hits() {
        let facts = Facts {
            cache_entries: Some(3),
            cache_bytes: Some(41 << 20),
            ..served()
        };
        let status = classify(&facts, None);
        let cache = status
            .conditions
            .iter()
            .find(|c| c.name == "PrefixCacheResident")
            .unwrap();
        assert!(cache.held);
        assert!(cache.reason.contains("residency, not hits"), "{}", cache.reason);
    }

    #[test]
    fn diff_reports_only_what_moved() {
        let before = classify(&served(), None);
        let after = classify(
            &Facts {
                stats: ok(400),
                ..served()
            },
            None,
        );
        let changes = diff(&before, &after);
        assert!(changes[0].contains("Ready -> Generating"));
        assert!(changes.iter().any(|c| c.starts_with("+Generating")));
        // The probe still answered, so its new latency is a measurement, not a
        // transition, and is reported as `~`.
        assert!(changes.iter().any(|c| c.starts_with("~EngineResponsive")));
        assert!(is_transition(&changes));
        assert!(!is_transition(&diff(&after, &after)));
    }

    #[test]
    fn a_locked_models_probe_does_not_unadmit_the_model() {
        // `/v1/models` queues behind the model lock, so timing out there during a
        // generation says nothing about whether a model is served.
        let facts = Facts {
            stats: ok(400),
            models: Probe::TimedOut(Duration::from_millis(300)),
            model_count: None,
            stats_model: Some("qwen3-0.6b-Q8_0".into()),
            ..served()
        };
        let status = classify(&facts, None);
        assert_eq!(status.phase, Phase::Generating);
        assert!(status
            .conditions
            .iter()
            .any(|c| c.name == "ModelsAdmitted" && c.held));
    }

    #[test]
    fn a_post_lock_timeout_during_a_generation_is_not_an_outage() {
        let facts = Facts {
            models: Probe::TimedOut(Duration::from_millis(300)),
            stats: Probe::TimedOut(Duration::from_millis(300)),
            model_count: None,
            stats_model: None,
            ..served()
        };
        let status = classify(&facts, None);
        assert_eq!(status.phase, Phase::Generating);
        for name in ["ModelsAdmitted", "EngineResponsive"] {
            let condition = status
                .conditions
                .iter()
                .find(|c| c.name == name)
                .unwrap();
            assert!(condition.held, "{name}: {}", condition.reason);
        }
    }

    #[test]
    fn the_same_timeout_on_a_dead_server_still_reads_as_unreachable() {
        let facts = Facts {
            health: Probe::Failed,
            stats: Probe::TimedOut(Duration::from_millis(300)),
            ..served()
        };
        let status = classify(&facts, None);
        assert_eq!(status.phase, Phase::Stopped);
        assert!(!status
            .conditions
            .iter()
            .any(|c| c.name == "EngineResponsive" && c.held));
    }

    #[test]
    fn an_unbudgeted_limit_is_uncapped_not_a_huge_number() {
        let facts = Facts {
            limit_bytes: Some(u64::MAX),
            ..served()
        };
        let status = classify(&facts, None);
        let headroom = status
            .conditions
            .iter()
            .find(|c| c.name == "MemoryHeadroom")
            .unwrap();
        assert!(headroom.held);
        assert!(headroom.reason.contains("no cap set"), "{}", headroom.reason);
        assert!(!measurements(&facts).contains("18446744073"));
    }

    #[test]
    fn an_exhausted_budget_drops_the_headroom_condition() {
        let facts = Facts {
            reserved_bytes: Some(96 << 30),
            ..served()
        };
        let status = classify(&facts, None);
        let headroom = status
            .conditions
            .iter()
            .find(|c| c.name == "MemoryHeadroom")
            .unwrap();
        assert!(!headroom.held);
    }
}
