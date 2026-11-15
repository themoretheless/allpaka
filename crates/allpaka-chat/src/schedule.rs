//! Persistent cron & interval scheduler (`schedule_{create,edit,list,view}`)
//! with anti-self-replication (`scheduled_origin == true` blocks schedule mutation).

use crate::ext::{default_settings, spawn_scheduled_session};
use crate::{bad_request, not_found, state_file, storage, ApiResult, App, Settings};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

const MAX_SCHEDULES: usize = 64;
const MIN_INTERVAL_SECS: u64 = 60;
const MAX_INTERVAL_SECS: u64 = 30 * 24 * 3600;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ScheduleEntry {
    pub id: String,
    pub name: String,
    pub project_id: String,
    pub schedule: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    pub created_at_ms: u64,
    #[serde(default)]
    pub next_run_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_session_id: Option<String>,
    #[serde(default)]
    pub run_count: u64,
}

fn default_enabled() -> bool {
    true
}

#[derive(Clone, Debug)]
enum ParsedSchedule {
    IntervalSecs(u64),
    Cron {
        min: Vec<bool>,
        hour: Vec<bool>,
        dom: Vec<bool>,
        month: Vec<bool>,
        dow: Vec<bool>,
    },
}

fn parse_cron_field(field: &str, min_v: u32, max_v: u32, wrap_7_to_0: bool) -> Result<Vec<bool>> {
    let mut table = vec![false; (max_v + 1) as usize];
    for part in field.split(',') {
        let part = part.trim();
        if part.is_empty() {
            bail!("Empty cron sub-expression in `{field}`");
        }
        let (range_part, step) = if let Some((l, r)) = part.split_once('/') {
            let s: u32 = r
                .parse()
                .with_context(|| format!("Invalid cron step `{r}`"))?;
            if s == 0 {
                bail!("Cron step must be >= 1");
            }
            (l, s)
        } else {
            (part, 1)
        };
        let (start, end) = if range_part == "*" {
            (min_v, max_v)
        } else if let Some((a, b)) = range_part.split_once('-') {
            let lo: u32 = a
                .parse()
                .with_context(|| format!("Invalid cron range start `{a}`"))?;
            let hi: u32 = b
                .parse()
                .with_context(|| format!("Invalid cron range end `{b}`"))?;
            if lo > hi {
                bail!("Cron range start exceeds end in `{part}`");
            }
            (lo, hi)
        } else {
            let exact: u32 = range_part
                .parse()
                .with_context(|| format!("Invalid cron value `{range_part}`"))?;
            (exact, exact)
        };
        if start < min_v || end > max_v {
            bail!("Cron value `{part}` out of allowed range {min_v}..={max_v}");
        }
        let mut v = start;
        while v <= end {
            let idx = if wrap_7_to_0 && v == 7 { 0 } else { v };
            table[idx as usize] = true;
            v = v.saturating_add(step);
        }
    }
    if !table.iter().any(|&x| x) {
        bail!("Cron field `{field}` matches no values");
    }
    Ok(table)
}

fn parse_schedule(raw: &str) -> Result<ParsedSchedule> {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed
        .strip_prefix("every ")
        .or_else(|| trimmed.strip_prefix("EVERY "))
    {
        let r = rest.trim().to_ascii_lowercase();
        let (num_str, mult) = if let Some(n) = r.strip_suffix('s') {
            (n, 1u64)
        } else if let Some(n) = r.strip_suffix('m') {
            (n, 60u64)
        } else if let Some(n) = r.strip_suffix('h') {
            (n, 3600u64)
        } else {
            bail!("Interval schedule must end in s, m, or h (e.g. `every 5m`)");
        };
        let num: u64 = num_str.trim().parse().context("Invalid interval count")?;
        let secs = num.saturating_mul(mult);
        if !(MIN_INTERVAL_SECS..=MAX_INTERVAL_SECS).contains(&secs) {
            bail!("Interval schedule must be between {MIN_INTERVAL_SECS}s and 30d");
        }
        return Ok(ParsedSchedule::IntervalSecs(secs));
    }

    let fields: Vec<&str> = trimmed.split_whitespace().collect();
    if fields.len() != 5 {
        bail!("Schedule must be `every <N>(s|m|h)` or a 5-field cron expression");
    }
    Ok(ParsedSchedule::Cron {
        min: parse_cron_field(fields[0], 0, 59, false)?,
        hour: parse_cron_field(fields[1], 0, 23, false)?,
        dom: parse_cron_field(fields[2], 1, 31, false)?,
        month: parse_cron_field(fields[3], 1, 12, false)?,
        dow: parse_cron_field(fields[4], 0, 7, true)?,
    })
}

/// Convert Unix epoch seconds to UTC `(month 1..=12, dom 1..=31, dow 0..=6 (0=Sun), hour, min)`.
fn unix_secs_to_utc_fields(unix_secs: i64) -> (usize, usize, usize, usize, usize) {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400) as usize;
    let hour = secs_of_day / 3600;
    let min = (secs_of_day % 3600) / 60;
    let dow = (days + 4).rem_euclid(7) as usize; // 1970-01-01 was Thursday (4)

    // Howard Hinnant's civil-from-days algorithm
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as usize;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as usize;
    (m, d, dow, hour, min)
}

pub(crate) fn next_fire_ms(spec: &str, now_ms: u64) -> Result<u64> {
    match parse_schedule(spec)? {
        ParsedSchedule::IntervalSecs(secs) => Ok(now_ms.saturating_add(secs * 1000)),
        ParsedSchedule::Cron {
            min,
            hour,
            dom,
            month,
            dow,
        } => {
            let start_min = (now_ms / 60_000) + 1;
            let max_scan = 366 * 24 * 60;
            for offset in 0..max_scan {
                let candidate_secs = ((start_min + offset) * 60) as i64;
                let (m, d, w, h, mn) = unix_secs_to_utc_fields(candidate_secs);
                if min[mn] && hour[h] && dom[d] && month[m] && dow[w] {
                    return Ok((candidate_secs as u64) * 1000);
                }
            }
            bail!("Cron expression `{spec}` did not match within 366 days");
        }
    }
}

fn validate_entry(e: &ScheduleEntry) -> Result<()> {
    if !storage::valid_id(&e.id) {
        bail!("Invalid schedule id `{}`", e.id);
    }
    if e.name.trim().is_empty() || e.name.chars().count() > 120 {
        bail!("Schedule name must be 1-120 characters");
    }
    if e.prompt.trim().is_empty() || e.prompt.len() > 16_000 {
        bail!("Schedule prompt must be 1-16000 bytes");
    }
    let _ = parse_schedule(&e.schedule)?;
    Ok(())
}

pub(crate) fn load_schedules(app: &App) -> Result<Vec<ScheduleEntry>> {
    state_file::load_list(app.data.as_path(), "schedules", "schedule", validate_entry)
}

fn save_schedules(app: &App, list: &[ScheduleEntry]) -> Result<()> {
    state_file::save_list(app.data.as_path(), "schedules", list)
}

pub(crate) fn schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "schedule_create",
                "description": "Create a persistent cron (`m h dom mon dow` UTC) or interval (`every 5m`) scheduled prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "schedule": { "type": "string" },
                        "prompt": { "type": "string" },
                        "profile": { "type": "string" },
                        "enabled": { "type": "boolean" }
                    },
                    "required": ["name", "schedule", "prompt"],
                    "additionalProperties": false
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "schedule_edit",
                "description": "Update an existing schedule's cron/interval expression, prompt, enabled status, or delete it (`delete: true`).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "schedule": { "type": "string" },
                        "prompt": { "type": "string" },
                        "enabled": { "type": "boolean" },
                        "delete": { "type": "boolean" }
                    },
                    "required": ["id"],
                    "additionalProperties": false
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "schedule_list",
                "description": "List all configured schedules.",
                "parameters": { "type": "object", "properties": {}, "additionalProperties": false }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "schedule_view",
                "description": "Inspect full details of a single schedule by ID.",
                "parameters": {
                    "type": "object",
                    "properties": { "id": { "type": "string" } },
                    "required": ["id"],
                    "additionalProperties": false
                }
            }
        }),
    ]
}

pub(crate) fn execute_tool(
    app: &App,
    settings: &Settings,
    scheduled_origin: bool,
    name: &str,
    args: &Value,
) -> Result<Value> {
    match name {
        "schedule_create" | "schedule_edit" => {
            if scheduled_origin {
                bail!("Scheduled sessions cannot create or edit schedules (anti-self-replication)");
            }
            if name == "schedule_create" {
                create_from_json(app, &settings.project_id, args)
            } else {
                edit_from_json(app, args)
            }
        }
        "schedule_list" => {
            let list = load_schedules(app)?;
            Ok(json!({ "schedules": list }))
        }
        "schedule_view" => {
            let id = args["id"].as_str().context("Missing `id`")?;
            let list = load_schedules(app)?;
            let item = list
                .into_iter()
                .find(|s| s.id == id)
                .with_context(|| format!("Schedule `{id}` not found"))?;
            Ok(json!({ "schedule": item }))
        }
        other => bail!("Unsupported schedule tool `{other}`"),
    }
}

fn create_from_json(app: &App, default_project: &str, args: &Value) -> Result<Value> {
    let mut list = load_schedules(app)?;
    if list.len() >= MAX_SCHEDULES {
        bail!("Maximum {MAX_SCHEDULES} schedules reached");
    }
    let now = storage::now_ms();
    let schedule_str = args["schedule"]
        .as_str()
        .context("Missing `schedule`")?
        .trim()
        .to_owned();
    let next = next_fire_ms(&schedule_str, now)?;
    let id = args["id"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("sch-{}", crate::evaluation::new_id()));
    if list.iter().any(|s| s.id == id) {
        bail!("Schedule `{id}` already exists");
    }
    let entry = ScheduleEntry {
        id,
        name: args["name"]
            .as_str()
            .context("Missing `name`")?
            .trim()
            .to_owned(),
        project_id: args["project_id"]
            .as_str()
            .unwrap_or(default_project)
            .to_owned(),
        schedule: schedule_str,
        prompt: args["prompt"]
            .as_str()
            .context("Missing `prompt`")?
            .to_owned(),
        profile: args["profile"].as_str().map(str::to_owned),
        enabled: args["enabled"].as_bool().unwrap_or(true),
        created_at_ms: now,
        next_run_ms: next,
        last_run_ms: None,
        last_session_id: None,
        run_count: 0,
    };
    validate_entry(&entry)?;
    list.push(entry.clone());
    save_schedules(app, &list)?;
    Ok(json!({ "schedule": entry }))
}

fn edit_from_json(app: &App, args: &Value) -> Result<Value> {
    let id = args["id"].as_str().context("Missing `id`")?;
    let mut list = load_schedules(app)?;
    if args["delete"].as_bool() == Some(true) {
        let before = list.len();
        list.retain(|s| s.id != id);
        if list.len() == before {
            bail!("Schedule `{id}` not found");
        }
        save_schedules(app, &list)?;
        return Ok(json!({ "deleted": id }));
    }
    let pos = list
        .iter()
        .position(|s| s.id == id)
        .with_context(|| format!("Schedule `{id}` not found"))?;
    let mut updated = list[pos].clone();
    if let Some(n) = args["name"].as_str() {
        updated.name = n.trim().to_owned();
    }
    if let Some(s) = args["schedule"].as_str() {
        updated.schedule = s.trim().to_owned();
        updated.next_run_ms = next_fire_ms(&updated.schedule, storage::now_ms())?;
    }
    if let Some(p) = args["prompt"].as_str() {
        updated.prompt = p.to_owned();
    }
    if let Some(en) = args["enabled"].as_bool() {
        updated.enabled = en;
    }
    validate_entry(&updated)?;
    list[pos] = updated.clone();
    save_schedules(app, &list)?;
    Ok(json!({ "schedule": updated }))
}

pub(crate) fn trigger_now(app: &App, id: &str) -> Result<Value> {
    let mut list = load_schedules(app)?;
    let pos = list
        .iter()
        .position(|s| s.id == id)
        .with_context(|| format!("Schedule `{id}` not found"))?;
    let now = storage::now_ms();
    let entry = &list[pos];
    let mut settings = default_settings(app, &entry.project_id);
    if let Some(profile_id) = &entry.profile {
        if let Ok(prof) = crate::bundles::resolve_profile(app, profile_id) {
            crate::bundles::apply_profile(&prof, &mut settings);
        }
    }
    let title = format!("Schedule: {}", entry.name);
    let session_id = spawn_scheduled_session(app, title, settings, entry.prompt.clone())?;
    let next = next_fire_ms(&entry.schedule, now)?;
    list[pos].last_run_ms = Some(now);
    list[pos].next_run_ms = next;
    list[pos].last_session_id = Some(session_id.clone());
    list[pos].run_count += 1;
    let updated = list[pos].clone();
    save_schedules(app, &list)?;
    Ok(json!({ "triggered": true, "session_id": session_id, "schedule": updated }))
}

pub(crate) fn spawn_worker(app: App) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(15));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let now = storage::now_ms();
            if let Ok(list) = load_schedules(&app) {
                for entry in list {
                    if entry.enabled && entry.next_run_ms > 0 && now >= entry.next_run_ms {
                        let _ = trigger_now(&app, &entry.id);
                    }
                }
            }
        }
    })
}

pub(crate) async fn list_api(State(app): State<App>) -> ApiResult<Value> {
    let schedules = load_schedules(&app).map_err(crate::internal)?;
    Ok(Json(json!({ "schedules": schedules })))
}

pub(crate) async fn create_api(
    State(app): State<App>,
    Json(body): Json<Value>,
) -> ApiResult<Value> {
    create_from_json(&app, "default", &body)
        .map(Json)
        .map_err(bad_request)
}

pub(crate) async fn delete_api(
    State(app): State<App>,
    AxumPath(sid): AxumPath<String>,
) -> ApiResult<Value> {
    let mut list = load_schedules(&app).map_err(crate::internal)?;
    let before = list.len();
    list.retain(|s| s.id != sid);
    if list.len() == before {
        return Err(not_found("Schedule not found"));
    }
    save_schedules(&app, &list).map_err(crate::internal)?;
    Ok(Json(json!({ "deleted": sid })))
}

pub(crate) async fn trigger_api(
    State(app): State<App>,
    AxumPath(sid): AxumPath<String>,
) -> ApiResult<Value> {
    trigger_now(&app, &sid).map(Json).map_err(bad_request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Project, Root};
    use crate::ext::test_app;

    #[test]
    fn cron_and_interval_next_fire_calculations() {
        let base_ms = 1_700_000_000_000u64;
        assert_eq!(
            next_fire_ms("every 5m", base_ms).unwrap(),
            base_ms + 300_000
        );
        assert!(next_fire_ms("every 10s", base_ms).is_err());

        let cron_next = next_fire_ms("0 */2 * * 1-5", base_ms).unwrap();
        assert!(cron_next > base_ms);
        let (_, _, dow, _, mn) = unix_secs_to_utc_fields((cron_next / 1000) as i64);
        assert_eq!(mn, 0);
        assert!((1..=5).contains(&dow));
    }

    #[test]
    fn scheduled_origin_blocks_schedule_create_and_edit() {
        let root_dir = std::env::temp_dir().join(format!("allpaka-sch-test-{}", storage::now_ms()));
        let ws = root_dir.join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let ws = ws.canonicalize().unwrap();
        let app = test_app(
            &root_dir.join("data"),
            &ws,
            vec![Project {
                id: "default".into(),
                name: "default".into(),
                roots: vec![Root {
                    alias: "workspace".into(),
                    path: ws.clone(),
                    writable: false,
                    repository: false,
                    ssh: None,
                }],
                instructions: String::new(),
            }],
        );
        let settings = default_settings(&app, "default");

        let created = execute_tool(
            &app,
            &settings,
            false,
            "schedule_create",
            &json!({ "name": "Hourly", "schedule": "every 1h", "prompt": "check status" }),
        )
        .unwrap();
        let sid = created["schedule"]["id"].as_str().unwrap().to_owned();

        assert!(execute_tool(
            &app,
            &settings,
            true,
            "schedule_create",
            &json!({ "name": "Nested", "schedule": "every 1h", "prompt": "loop" }),
        )
        .is_err());
        assert!(execute_tool(
            &app,
            &settings,
            true,
            "schedule_edit",
            &json!({ "id": sid, "enabled": false }),
        )
        .is_err());
        assert_eq!(
            execute_tool(&app, &settings, false, "schedule_list", &json!({})).unwrap()["schedules"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let _ = std::fs::remove_dir_all(&root_dir);
    }
}
