//! Swarm mode: a wave of parallel agents on one task, then one merged answer.
//!
//! Shape of a turn:
//!   1. fan-out   — every member gets its own persona, provider and model plus a
//!                  read-only tool loop, and streams into its own report slot;
//!   2. fan-in    — one synthesizer call merges the reports into a MASTER-style
//!                  answer (deduplicated, conflicts kept explicit);
//!   3. critic    — optional adversarial pass over the merged draft.
//!
//! Members never write files: they run with `Mode::Chat` schemas regardless of the
//! session's write permission, and only `list_files`/`read_file` are accepted.
//! Reports live in `Message::swarm`, so one turn stays one message in history.

use crate::provider::{self, Provider};
use crate::types::*;
use crate::{context, persist, tools, App, SharedSession, TurnOutcome};
use anyhow::{bail, Context, Result};
use futures_util::future::join_all;
use serde_json::{json, Value};

/// Bounds enforced for every swarm turn.
pub const MIN_MEMBERS: usize = 2;
pub const MAX_MEMBERS: usize = 6;
pub const MAX_ROUNDS: u8 = 2;
pub const MAX_STEPS_PER_MEMBER: usize = 8;
pub const MIN_REPORT_BYTES: usize = 1000;
pub const MAX_REPORT_BYTES: usize = 24000;
/// Members started at the same time. Keeps provider rate limits survivable.
const MAX_PARALLEL: usize = 3;
/// Bytes per peer report quoted into a wave-2 prompt.
const PEER_DIGEST_BYTES: usize = 2000;
/// Bytes per report quoted into the critic prompt.
const CRITIC_REPORT_BYTES: usize = 1500;

/// Structural validation. Provider existence and keys are checked by the server,
/// which is the only place that knows the connection list.
pub fn validate(config: &SwarmConfig) -> Result<()> {
    if config.members.len() < MIN_MEMBERS || config.members.len() > MAX_MEMBERS {
        bail!(
            "Swarm mode needs {MIN_MEMBERS}–{MAX_MEMBERS} members, got {}",
            config.members.len()
        );
    }
    if !(1..=MAX_ROUNDS).contains(&config.rounds) {
        bail!("Swarm waves must be between 1 and {MAX_ROUNDS}");
    }
    if !(1..=MAX_STEPS_PER_MEMBER).contains(&config.max_steps_per_member) {
        bail!("Swarm steps per member must be between 1 and {MAX_STEPS_PER_MEMBER}");
    }
    if !(MIN_REPORT_BYTES..=MAX_REPORT_BYTES).contains(&config.report_bytes) {
        bail!("Swarm report budget must be between {MIN_REPORT_BYTES} and {MAX_REPORT_BYTES} bytes");
    }
    if config.synthesis_provider.chars().any(char::is_control)
        || config.synthesis_model.chars().any(char::is_control)
    {
        bail!("Synthesis provider and model must be printable text");
    }
    let mut labels = std::collections::HashSet::new();
    for member in &config.members {
        let label = member.label.trim();
        if label.is_empty() || label.chars().count() > 60 || label.chars().any(char::is_control) {
            bail!("Every swarm member needs a name of 1–60 printable characters");
        }
        if !labels.insert(label.to_lowercase()) {
            bail!("Swarm member names must be unique: {label}");
        }
        if member.provider.trim().is_empty() {
            bail!("Swarm member {label} has no provider");
        }
        if member.model.trim().is_empty() || member.model.len() > 200 {
            bail!("Swarm member {label} needs a model ID");
        }
        if member.role.chars().count() > 4000
            || member
                .role
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t')
        {
            bail!("Swarm member {label}: persona must be at most 4000 characters");
        }
    }
    Ok(())
}

/// Run one swarm turn for the session's queued user message.
pub async fn run(
    app: &App,
    shared: &SharedSession,
    settings: &Settings,
    project: &context::Project,
    system: &str,
) -> Result<TurnOutcome> {
    let config = settings.swarm.clone();
    validate(&config)?;
    let (members, synth) = resolve(app, settings, &config)?;
    let read_only = {
        let mut member_view = settings.clone();
        member_view.mode = Mode::Chat;
        member_view.allow_writes = false;
        tools::schemas(&member_view, false)
    };

    let (index, brief) = prepare(shared, &members);
    persist(app, shared);
    // Cost is visible up front: one turn is members × waves + merge (+critic) requests.
    {
        let planned = config.request_count();
        let mut state = shared.lock().unwrap();
        state.notice = Some(format!(
            "Swarm: участников {count} × волн {rounds}{critic} — до {planned} запросов к провайдерам на этот ход.",
            count = config.members.len(),
            rounds = config.rounds.max(1),
            critic = if config.critic { " + критик" } else { "" },
            planned = planned,
        ));
    }

    let mut prompts: Vec<String> = members
        .iter()
        .map(|(member, _)| wave_one_prompt(member, &config, &brief))
        .collect();
    let mut total_usage = Value::Null;
    let rounds = config.rounds.max(1);

    for round in 1..=rounds {
        let mut start = 0;
        while start < members.len() {
            let end = (start + MAX_PARALLEL).min(members.len());
            let mut futures = Vec::new();
            for slot in start..end {
                let (member, provider) = &members[slot];
                futures.push(run_member(
                    app,
                    shared,
                    project,
                    &member.label,
                    provider,
                    member_settings(settings, member, &config),
                    member_system(system, member, &config, round),
                    &read_only,
                    round,
                    prompts[slot].clone(),
                    slot,
                    index,
                ));
            }
            let results = join_all(futures).await;
            for (offset, result) in results.into_iter().enumerate() {
                match result {
                    Ok(outcome) => total_usage = merge_usage(total_usage, outcome.usage),
                    Err(err) => {
                        let slot = start + offset;
                        fail(shared, index, slot, &format!("{err:#}"));
                    }
                }
            }
            start = end;
        }
        if round < rounds {
            prompts = next_round_prompts(shared, index, &members, &config, &brief);
        }
    }

    let reports = collect_reports(shared, index);
    if reports.iter().all(|r| r.content.trim().is_empty()) {
        finish(
            app,
            shared,
            index,
            total_usage,
            Some(
                "Ни один участник Swarm не вернул отчёт. Проверьте ключи, доступ к моделям и лимиты — результат не выдуман."
                    .into(),
            ),
        );
        bail!("Ни один участник Swarm не вернул отчёт");
    }

    let (synth_usage, notice, truncated) =
        synthesize(app, shared, index, settings, &config, &synth, &brief, &reports).await?;
    total_usage = merge_usage(total_usage, synth_usage);

    finish(app, shared, index, total_usage, notice);
    if truncated {
        return Ok(TurnOutcome::TokenLimit);
    }
    Ok(TurnOutcome::Complete)
}

/// Re-run one member of the last Swarm turn and rebuild the answer from all the
/// reports that stand afterwards. The member is taken from the stored report, not
/// from the current roster, so editing the roster between turns does not hide the
/// retry; only its persona is re-read from the roster, and a member that is no
/// longer there gets the generic role.
pub async fn retry(
    app: &App,
    shared: &SharedSession,
    settings: &Settings,
    project: &context::Project,
    system: &str,
    label: &str,
) -> Result<TurnOutcome> {
    let config = settings.swarm.clone();
    let target = take_report(shared, label, &config)?;
    let provider = {
        let providers = app.providers.lock().unwrap().clone();
        providers
            .iter()
            .find(|p| p.id == target.member.provider)
            .cloned()
    }
    .with_context(|| {
        format!(
            "Участник {}: неизвестный провайдер {}",
            target.member.label, target.member.provider
        )
    })?;
    let read_only = {
        let mut member_view = settings.clone();
        member_view.mode = Mode::Chat;
        member_view.allow_writes = false;
        tools::schemas(&member_view, false)
    };
    let prompt = if target.round > 1 {
        wave_two_prompt(&target.member, &target.own, &peer_digests(shared, target.index, target.slot), &config)
    } else {
        wave_one_prompt(&target.member, &config, &target.brief)
    };
    let outcome = run_member(
        app,
        shared,
        project,
        &target.member.label,
        &provider,
        member_settings(settings, &target.member, &config),
        member_system(system, &target.member, &config, target.round),
        &read_only,
        target.round,
        format!("{prompt}{RETRY_NOTE}"),
        target.slot,
        target.index,
    )
    .await?;
    let mut total_usage = outcome.usage;

    let reports = collect_reports(shared, target.index);
    if reports.iter().all(|r| r.content.trim().is_empty()) {
        finish(
            app,
            shared,
            target.index,
            total_usage,
            Some("Повтор не дал отчёта, и остальные участники пусты — результат не выдуман.".into()),
        );
        bail!("Повтор участника не вернул отчёт");
    }
    // The synthesis streams into the message, so the previous MASTER has to go
    // first; otherwise the rebuild would read as an appended second answer.
    set_content(shared, target.index, "");
    let (synth_usage, notice, truncated) = synthesize(
        app,
        shared,
        target.index,
        settings,
        &config,
        &synth_provider(app, settings, &config)?,
        &target.brief,
        &reports,
    )
    .await?;
    total_usage = merge_usage(total_usage, synth_usage);
    finish(app, shared, target.index, total_usage, notice);
    if truncated {
        return Ok(TurnOutcome::TokenLimit);
    }
    Ok(TurnOutcome::Complete)
}

/// The synthesizer connection, resolved the same way `run` resolves it.
fn synth_provider(app: &App, settings: &Settings, config: &SwarmConfig) -> Result<Provider> {
    let name = if config.synthesis_provider.trim().is_empty() {
        settings.provider.clone()
    } else {
        config.synthesis_provider.clone()
    };
    let providers = app.providers.lock().unwrap().clone();
    providers
        .iter()
        .find(|p| p.id == name)
        .cloned()
        .with_context(|| format!("Синтез: неизвестный провайдер {name}"))
}

/// Merge the reports that exist right now into the Swarm message, then run the
/// critic pass over that draft when it is enabled. Returns the usage of these
/// requests, the notice for the caller and whether the merge hit a token limit.
async fn synthesize(
    app: &App,
    shared: &SharedSession,
    index: usize,
    settings: &Settings,
    config: &SwarmConfig,
    synth: &Provider,
    brief: &str,
    reports: &[ReportView],
) -> Result<(Value, Option<String>, bool)> {
    let merge_cfg = merge_settings(settings, config);
    let (merged, mut usage) = merge_pass(
        app,
        shared,
        index,
        synth,
        &merge_cfg,
        "You are the synthesizer of a swarm. Merge peer reports into one answer. Never invent findings nobody reported.",
        merge_prompt(brief, reports, config),
        MergeOutput::Stream,
    )
    .await?;
    let mut notice = failure_notice(reports, reports.len());
    if merged.truncated {
        return Ok((usage, notice, true));
    }
    if config.critic {
        let draft = message_content(shared, index);
        let (critic, critic_usage) = merge_pass(
            app,
            shared,
            index,
            synth,
            &merge_cfg,
            "You are an adversarial reviewer of a merged swarm result. Return only the corrected final text.",
            critic_prompt(brief, reports, &draft, config),
            MergeOutput::Buffered,
        )
        .await?;
        usage = merge_usage(usage, critic_usage);
        if critic.truncated {
            notice = Some(match notice {
                Some(text) => format!("{text} Критик-проход прерван лимитом токенов: показан черновик синтеза."),
                None => "Критик-проход прерван лимитом токенов: показан черновик синтеза.".into(),
            });
        } else if critic.content.trim().is_empty() {
            notice = Some(match notice {
                Some(text) => format!("{text} Критик вернул пустой ответ: показан черновик синтеза."),
                None => "Критик вернул пустой ответ: показан черновик синтеза.".into(),
            });
        } else {
            set_content(shared, index, &critic.content);
        }
    }
    Ok((usage, notice, false))
}

/// One member of the last Swarm turn, taken out of the message and reset for a
/// re-run: whatever the previous attempt produced is gone before the new one
/// streams, so a retried report never shows text from two attempts.
fn take_report(
    shared: &SharedSession,
    label: &str,
    config: &SwarmConfig,
) -> Result<RetryTarget> {
    let mut state = shared.lock().unwrap();
    let index = state
        .messages
        .iter()
        .rposition(|m| !m.swarm.is_empty())
        .context("В этом разговоре ещё не было Swarm-хода")?;
    let brief = state
        .messages[..index]
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let slot = state.messages[index]
        .swarm
        .iter()
        .position(|r| r.label.trim().eq_ignore_ascii_case(label.trim()))
        .with_context(|| format!("Участника «{label}» нет в последнем Swarm-ходе"))?;
    let previous = state.messages[index].swarm[slot].clone();
    let own = clip(&previous.content, config.report_bytes);
    let member = SwarmMember {
        label: previous.label.clone(),
        role: config
            .members
            .iter()
            .find(|m| m.label.trim().eq_ignore_ascii_case(previous.label.trim()))
            .map(|m| m.role.clone())
            .unwrap_or_default(),
        provider: previous.provider.clone(),
        model: previous.model.clone(),
    };
    let report = &mut state.messages[index].swarm[slot];
    report.content.clear();
    report.error = None;
    report.status = MemberStatus::Queued;
    report.step = 0;
    Ok(RetryTarget {
        index,
        slot,
        round: previous.round.max(1),
        brief,
        own,
        member,
    })
}

struct RetryTarget {
    index: usize,
    slot: usize,
    round: u8,
    brief: String,
    /// The member's own previous report, already clipped to the report budget.
    own: String,
    member: SwarmMember,
}

/// What the rest of the wave looks like for one member: the same digests the
/// next-wave prompt gets, minus the member's own.
fn peer_digests(shared: &SharedSession, index: usize, slot: usize) -> String {
    collect_reports(shared, index)
        .iter()
        .enumerate()
        .filter(|(other, _)| *other != slot)
        .map(|(_, r)| {
            format!(
                "### {label} ({provider}/{model})\n{content}",
                label = r.label,
                provider = r.provider,
                model = r.model,
                content = clip(&r.content, PEER_DIGEST_BYTES),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

const RETRY_NOTE: &str = "\n\nЭто повтор: прошлый заход этого участника не дошёл до синтезатора. \
     Выдай отчёт заново и целиком, не ссылаясь на прежнюю попытку.";

struct MemberOutcome {
    usage: Value,
}

#[derive(Clone)]
struct ReportView {
    label: String,
    provider: String,
    model: String,
    /// Wave the reported text comes from; a wave-2 report supersedes the first.
    round: u8,
    content: String,
    error: Option<String>,
}

enum MergeOutput {
    /// Stream into the assistant message as the synthesizer writes.
    Stream,
    /// Keep the provider output local; the caller decides whether to apply it.
    Buffered,
}

/// Resolve member and synthesizer connections from the live provider list.
fn resolve(
    app: &App,
    settings: &Settings,
    config: &SwarmConfig,
) -> Result<(Vec<(SwarmMember, Provider)>, Provider)> {
    let providers = app.providers.lock().unwrap().clone();
    let mut members = Vec::with_capacity(config.members.len());
    for member in &config.members {
        let provider = providers
            .iter()
            .find(|p| p.id == member.provider)
            .cloned()
            .with_context(|| {
                format!(
                    "Участник {}: неизвестный провайдер {}",
                    member.label, member.provider
                )
            })?;
        members.push((member.clone(), provider));
    }
    let synthesis = if config.synthesis_provider.trim().is_empty() {
        settings.provider.clone()
    } else {
        config.synthesis_provider.clone()
    };
    let synth = providers
        .iter()
        .find(|p| p.id == synthesis)
        .cloned()
        .with_context(|| format!("Синтез: неизвестный провайдер {synthesis}"))?;
    Ok((members, synth))
}

fn member_settings(settings: &Settings, member: &SwarmMember, config: &SwarmConfig) -> Settings {
    let mut member_settings = settings.clone();
    member_settings.provider = member.provider.clone();
    member_settings.model = member.model.clone();
    // Members read the context and report; they never write, whatever the session allows.
    member_settings.mode = Mode::Chat;
    member_settings.allow_writes = false;
    member_settings.json_mode = false;
    member_settings.max_steps = config.max_steps_per_member.clamp(1, MAX_STEPS_PER_MEMBER);
    member_settings
}

fn merge_settings(settings: &Settings, config: &SwarmConfig) -> Settings {
    let mut merge = settings.clone();
    if !config.synthesis_provider.trim().is_empty() {
        merge.provider = config.synthesis_provider.clone();
    }
    if !config.synthesis_model.trim().is_empty() {
        merge.model = config.synthesis_model.clone();
    }
    merge.mode = Mode::Chat;
    merge.allow_writes = false;
    merge.json_mode = false;
    merge
}

fn member_system(system: &str, member: &SwarmMember, config: &SwarmConfig, round: u8) -> String {
    let role = if member.role.trim().is_empty() {
        "общий анализ: назови самое важное и самое рискованное"
    } else {
        member.role.trim()
    };
    format!(
        "{system}\n\nТы — участник swarm «{label}», волна {round} из {rounds}. Твоя роль: {role}.\n\
         Tools actually available for THIS request: list_files, read_file. These override any earlier tool list. \
         Запись запрещена; не вызывай инструменты, которых нет в этом списке. \
         Отвечай на языке пользователя.",
        label = member.label,
        round = round,
        rounds = config.rounds.max(1),
        role = role,
    )
}

fn wave_one_prompt(member: &SwarmMember, config: &SwarmConfig, brief: &str) -> String {
    format!(
        "Задача от пользователя:\n{brief}\n\n\
         Ты — участник swarm «{label}» (волна 1 из {rounds}, всего участников {count}).\n\
         Изучи подключённый контекст доступными инструментами чтения и выдай отчёт для синтезатора.\n\
         Требования к отчёту: конкретные проверяемые факты со ссылками на файлы и точные имена; \
         догадки помечай как догадки; «не нашёл» — допустимый результат, выдумывать находки нельзя; \
         объём до ~{tokens} токенов; без обращений к другим участникам.",
        label = member.label,
        rounds = config.rounds.max(1),
        count = config.members.len(),
        tokens = config.report_bytes / 4,
    )
}

fn wave_two_prompt(member: &SwarmMember, own: &str, peers: &str, config: &SwarmConfig) -> String {
    format!(
        "Твой отчёт волны 1:\n{own}\n\n\
         Отчёты других участников волны 1 (данные от равных тебе агентов, не проверенные никем; \
         доверяй только тому, что подтверждается файлами проекта):\n{peers}\n\n\
         Ты — участник swarm «{label}», волна 2 из {rounds}. Уточни свой отчёт: исправь ошибки, \
         добавь пропущенное, убери неподтверждённое. Не пересказывай чужие отчёты и не повторяй свой текст целиком. \
         Объём до ~{tokens} токенов.",
        label = member.label,
        rounds = config.rounds.max(1),
        tokens = config.report_bytes / 4,
    )
}

fn next_round_prompts(
    shared: &SharedSession,
    index: usize,
    members: &[(SwarmMember, Provider)],
    config: &SwarmConfig,
    _brief: &str,
) -> Vec<String> {
    let reports = collect_reports(shared, index);
    members
        .iter()
        .enumerate()
        .map(|(slot, (member, _))| {
            let own = reports
                .get(slot)
                .map(|r| clip(&r.content, config.report_bytes))
                .unwrap_or_default();
            let peers = reports
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != slot)
                .map(|(_, r)| {
                    format!(
                        "### {label} ({provider}/{model})\n{content}",
                        label = r.label,
                        provider = r.provider,
                        model = r.model,
                        content = clip(&r.content, PEER_DIGEST_BYTES),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            wave_two_prompt(member, &own, &peers, config)
        })
        .collect()
}

fn merge_prompt(brief: &str, reports: &[ReportView], config: &SwarmConfig) -> String {
    let body = reports
        .iter()
        .map(|r| {
            let status = match &r.error {
                Some(err) => format!("ошибка: {err}"),
                None => "отчёт получен".into(),
            };
            format!(
                "### {label} ({provider}/{model}) · волна {round} — {status}\n{content}",
                label = r.label,
                provider = r.provider,
                model = r.model,
                round = r.round,
                status = status,
                content = clip(&r.content, config.report_bytes),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "Задача, над которой работали участники:\n{brief}\n\n\
         Отчёты {count} независимых агентов:\n{body}\n\n\
         Собери из них один итоговый ответ (MASTER) в таком порядке:\n\
         1) Сводка — 3–7 строк: что найдено и насколько задача закрыта.\n\
         2) Топ — самые важные пункты, только те, что подтверждены хотя бы одним отчётом, \
         отсортированные по влиянию, с указанием файлов и участника-источника в скобках.\n\
         3) Следующие шаги — до 5 пунктов, в порядке выполнения.\n\
         4) Конфликты и неподтверждённое — явно, если такие есть; иначе пропусти раздел.\n\
         Правила: дубликаты слить; не добавлять пунктов, которых не было в отчётах; \
         не выдавать догадку за факт; ошибки участников перечислять честно. Отвечай на языке пользователя.",
        count = reports.len(),
        brief = brief,
        body = body,
    )
}

fn critic_prompt(brief: &str, reports: &[ReportView], draft: &str, config: &SwarmConfig) -> String {
    let body = reports
        .iter()
        .map(|r| {
            format!(
                "### {label} · волна {round}\n{content}",
                label = r.label,
                round = r.round,
                content = clip(&r.content, CRITIC_REPORT_BYTES.max(config.report_bytes / 4)),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "Задача:\n{brief}\n\nОтчёты участников:\n{body}\n\nЧерновик синтеза:\n{draft}\n\n\
         Ты — критик swarm-результата. Удали всё, что не подтверждается отчётами, пометь противоречия, \
         уточни формулировки, сохрани структуру (сводка, топ, следующие шаги, конфликты). \
         Не добавляй новых пунктов. Верни только исправленный итоговый текст, без пояснений о правках.",
        brief = brief,
        body = body,
        draft = draft,
    )
}

fn failure_notice(reports: &[ReportView], members: usize) -> Option<String> {
    let failed = reports.iter().filter(|r| r.error.is_some()).count();
    (failed > 0).then(|| {
        format!("Swarm: {failed} из {members} участников не ответили; их отчёты помечены ошибкой и названы в итоге.")
    })
}

fn prepare(shared: &SharedSession, members: &[(SwarmMember, Provider)]) -> (usize, String) {
    let mut state = shared.lock().unwrap();
    let mut brief = state
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    for text in std::mem::take(&mut state.steering) {
        brief.push_str(&format!("\n\nUser steering for the current task: {text}"));
    }
    let index = state.messages.len();
    let mut message = Message::text("assistant", "");
    message.swarm = members
        .iter()
        .map(|(member, _)| SwarmReport {
            label: member.label.clone(),
            provider: member.provider.clone(),
            model: member.model.clone(),
            round: 1,
            status: MemberStatus::Queued,
            step: 0,
            content: String::new(),
            error: None,
        })
        .collect();
    state.messages.push(message);
    state.step = 1;
    (index, brief)
}

#[allow(clippy::too_many_arguments)]
async fn run_member(
    app: &App,
    shared: &SharedSession,
    project: &context::Project,
    label: &str,
    provider: &Provider,
    member_settings: Settings,
    system: String,
    schemas: &[Value],
    round: u8,
    prompt: String,
    slot: usize,
    index: usize,
) -> Result<MemberOutcome> {
    let mut history = vec![Message::text("user", prompt)];
    let mut usage = Value::Null;
    let steps = member_settings.max_steps.max(1);
    for step in 1..=steps {
        note(shared, index, slot, round, MemberStatus::Running, Some(step));
        if step > 1 {
            append(shared, index, slot, "\n\n");
        }
        let copy = shared.clone();
        let save_app = app.clone();
        let mut last_save = std::time::Instant::now();
        let (message, step_usage) = provider::generate(
            &app.client,
            provider,
            &member_settings,
            &system,
            &history,
            schemas,
            move |text, _reasoning| {
                if text.is_empty() {
                    return;
                }
                {
                    let mut state = copy.lock().unwrap();
                    if let Some(report) = state
                        .messages
                        .get_mut(index)
                        .and_then(|m| m.swarm.get_mut(slot))
                    {
                        report.content.push_str(&text);
                    }
                }
                if last_save.elapsed().as_secs() >= 2 {
                    persist(&save_app, &copy);
                    last_save = std::time::Instant::now();
                }
            },
        )
        .await
        .with_context(|| format!("участник {label}"))?;
        usage = merge_usage(usage, step_usage);
        if message.truncated {
            note(shared, index, slot, round, MemberStatus::Error, None);
            fail(shared, index, slot, "Достигнут лимит токенов: отчёт неполный");
            persist(app, shared);
            return Ok(MemberOutcome { usage });
        }
        if message.tool_calls.is_empty() {
            break;
        }
        history.push(message.clone());
        for call in &message.tool_calls {
            let value = run_tool(app, project, &member_settings, call);
            history.push(Message {
                role: "tool".into(),
                content: value.to_string(),
                tool_call_id: call["id"].as_str().map(str::to_owned),
                ..Message::default()
            });
        }
    }
    note(shared, index, slot, round, MemberStatus::Done, None);
    persist(app, shared);
    Ok(MemberOutcome { usage })
}

/// Read-only tool execution for swarm members. Anything else is refused, not
/// attempted: the member's schema does not offer it, so a call for it is a lie.
fn run_tool(app: &App, project: &context::Project, settings: &Settings, call: &Value) -> Value {
    let name = call["function"]["name"].as_str().unwrap_or("");
    if !tools::reads(name) {
        return json!({
            "error": format!("В Swarm-режиме доступны только list_files и read_file; вызов {name} отклонён")
        });
    }
    let args: Value = match call["function"]["arguments"]
        .as_str()
        .map(serde_json::from_str::<Value>)
    {
        Some(Ok(value)) => value,
        _ => return json!({"error": "Malformed tool arguments"}),
    };
    if name == "list_files" && args["path"] == "." {
        return json!({"contexts": project.roots.iter().map(|r| json!({"alias":r.alias,"writable":r.writable,"repository":r.repository})).collect::<Vec<_>>()});
    }
    let (root, path) = match project.root(args["path"].as_str().unwrap_or(""), false) {
        Ok(pair) => pair,
        Err(err) => return json!({"error": err.to_string()}),
    };
    let candidate = root.join(&path);
    let canonical = candidate.canonicalize().unwrap_or(candidate);
    if canonical.starts_with(app.data.as_ref()) {
        return json!({"error": "Conversation storage is not available to tools"});
    }
    let mut args = args.clone();
    args["path"] = json!(path);
    match tools::execute(root, settings, name, &args) {
        Ok(value) => value,
        Err(err) => json!({"error": err.to_string()}),
    }
}

#[allow(clippy::too_many_arguments)]
async fn merge_pass(
    app: &App,
    shared: &SharedSession,
    index: usize,
    provider: &Provider,
    settings: &Settings,
    system: &str,
    prompt: String,
    output: MergeOutput,
) -> Result<(Message, Value)> {
    let messages = [Message::text("user", prompt)];
    match output {
        MergeOutput::Buffered => {
            provider::generate(
                &app.client,
                provider,
                settings,
                system,
                &messages,
                &[],
                |_, _| {},
            )
            .await
        }
        MergeOutput::Stream => {
            let copy = shared.clone();
            let save_app = app.clone();
            let mut last_save = std::time::Instant::now();
            provider::generate(
                &app.client,
                provider,
                settings,
                system,
                &messages,
                &[],
                move |text, _reasoning| {
                    if text.is_empty() {
                        return;
                    }
                    {
                        let mut state = copy.lock().unwrap();
                        if let Some(message) = state.messages.get_mut(index) {
                            message.content.push_str(&text);
                        }
                    }
                    if last_save.elapsed().as_secs() >= 2 {
                        persist(&save_app, &copy);
                        last_save = std::time::Instant::now();
                    }
                },
            )
            .await
        }
    }
}

fn collect_reports(shared: &SharedSession, index: usize) -> Vec<ReportView> {
    let state = shared.lock().unwrap();
    state
        .messages
        .get(index)
        .map(|m| {
            m.swarm
                .iter()
                .map(|r| ReportView {
                    label: r.label.clone(),
                    provider: r.provider.clone(),
                    model: r.model.clone(),
                    round: r.round,
                    content: r.content.trim().to_owned(),
                    error: r.error.clone(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn message_content(shared: &SharedSession, index: usize) -> String {
    let state = shared.lock().unwrap();
    state
        .messages
        .get(index)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

fn set_content(shared: &SharedSession, index: usize, text: &str) {
    let mut state = shared.lock().unwrap();
    if let Some(message) = state.messages.get_mut(index) {
        message.content = text.to_owned();
    }
}

fn note(
    shared: &SharedSession,
    index: usize,
    slot: usize,
    round: u8,
    status: MemberStatus,
    step: Option<usize>,
) {
    let mut state = shared.lock().unwrap();
    if let Some(report) = state
        .messages
        .get_mut(index)
        .and_then(|m| m.swarm.get_mut(slot))
    {
        report.round = round;
        report.status = status;
        if let Some(step) = step {
            report.step = step;
        }
    }
}

fn append(shared: &SharedSession, index: usize, slot: usize, text: &str) {
    let mut state = shared.lock().unwrap();
    if let Some(report) = state
        .messages
        .get_mut(index)
        .and_then(|m| m.swarm.get_mut(slot))
    {
        report.content.push_str(text);
    }
}

fn fail(shared: &SharedSession, index: usize, slot: usize, message: &str) {
    let mut state = shared.lock().unwrap();
    if let Some(report) = state
        .messages
        .get_mut(index)
        .and_then(|m| m.swarm.get_mut(slot))
    {
        report.status = MemberStatus::Error;
        if report.error.is_none() {
            report.error = Some(message.to_owned());
        }
    }
}

fn finish(
    app: &App,
    shared: &SharedSession,
    index: usize,
    usage: Value,
    notice: Option<String>,
) {
    {
        let mut state = shared.lock().unwrap();
        state.usage = usage;
        if let Some(message) = state.messages.get_mut(index) {
            for report in &mut message.swarm {
                if report.status.unfinished() {
                    report.status = MemberStatus::Error;
                    if report.error.is_none() {
                        report.error = Some("Отчёт прерван".into());
                    }
                }
            }
        }
        if let Some(text) = notice {
            state.notice = Some(text);
        } else {
            // The start-of-turn cost note is progress information, not a result.
            state.notice = None;
        }
    }
    persist(app, shared);
}

/// Sum numeric usage fields across the requests of one swarm turn. The last
/// provider/model identity wins so the UI can still name the model that merged.
fn merge_usage(mut total: Value, next: Value) -> Value {
    let Some(incoming) = next.as_object() else {
        return total;
    };
    if !total.is_object() {
        total = json!({});
    }
    let fields = total.as_object_mut().unwrap();
    for (key, value) in incoming {
        if let Some(number) = value.as_f64() {
            let sum = fields.get(key).and_then(Value::as_f64).unwrap_or(0.0) + number;
            fields.insert(
                key.clone(),
                if sum.fract() == 0.0 {
                    json!(sum as i64)
                } else {
                    json!(sum)
                },
            );
        } else if value.is_string() {
            fields.insert(key.clone(), value.clone());
        }
    }
    total
}

/// Cut a report to `bytes` without splitting a UTF-8 character.
fn clip(text: &str, bytes: usize) -> String {
    if text.len() <= bytes {
        return text.to_owned();
    }
    let mut end = bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[сокращено]", &text[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(label: &str) -> SwarmMember {
        SwarmMember {
            label: label.into(),
            role: "проверь риски".into(),
            provider: "local".into(),
            model: "qwen3".into(),
        }
    }
    fn config(members: usize) -> SwarmConfig {
        SwarmConfig {
            members: (0..members).map(|i| member(&format!("m{i}"))).collect(),
            ..SwarmConfig::default()
        }
    }

    #[test]
    fn bounds_reject_too_few_members_and_odd_rounds() {
        assert!(validate(&config(1)).is_err());
        assert!(validate(&config(2)).is_ok());
        assert!(validate(&config(MAX_MEMBERS)).is_ok());
        assert!(validate(&config(MAX_MEMBERS + 1)).is_err());
        let mut rounds = config(2);
        rounds.rounds = MAX_ROUNDS + 1;
        assert!(validate(&rounds).is_err());
        let mut steps = config(2);
        steps.max_steps_per_member = 0;
        assert!(validate(&steps).is_err());
        let mut budget = config(2);
        budget.report_bytes = MIN_REPORT_BYTES - 1;
        assert!(budget.report_bytes < MIN_REPORT_BYTES);
        assert!(validate(&budget).is_err());
    }

    #[test]
    fn member_names_are_unique_and_required() {
        let mut duplicate = config(2);
        duplicate.members[1].label = "M0".into();
        assert!(validate(&duplicate).is_err());
        let mut blank = config(2);
        blank.members[0].label = "  ".into();
        assert!(validate(&blank).is_err());
        let mut no_model = config(2);
        no_model.members[0].model = String::new();
        assert!(validate(&no_model).is_err());
    }

    #[test]
    fn request_count_matches_the_documented_formula() {
        let mut c = config(4);
        assert_eq!(c.request_count(), 4 + 1);
        c.rounds = 2;
        assert_eq!(c.request_count(), 8 + 1);
        c.critic = true;
        assert_eq!(c.request_count(), 8 + 2);
    }

    #[test]
    fn usage_fields_sum_and_identity_is_last_wins() {
        let total = merge_usage(
            json!({"prompt_tokens":10,"completion_tokens":5,"model":"a"}),
            json!({"prompt_tokens":4,"completion_tokens":3,"model":"b","cost":0.25}),
        );
        assert_eq!(total["prompt_tokens"], 14);
        assert_eq!(total["completion_tokens"], 8);
        assert_eq!(total["model"], "b");
        assert_eq!(total["cost"], 0.25);
        assert_eq!(merge_usage(Value::Null, json!(null)), Value::Null);
    }

    #[test]
    fn clip_never_splits_a_character() {
        let text = "привет мир";
        assert_eq!(clip(text, 100), text);
        // 7 lands in the middle of a two-byte character; the cut moves to 6.
        assert_eq!(clip(text, 7), "при\n[сокращено]");
        // 12 lands exactly on the boundary after "т"; the cut stays put.
        assert_eq!(clip(text, 12), "привет\n[сокращено]");
    }

    #[test]
    fn merge_prompt_names_members_and_forbids_invented_items() {
        let reports = vec![
            ReportView {
                label: "safety".into(),
                provider: "openai".into(),
                model: "gpt".into(),
                round: 2,
                content: "нашёл дыру в доступе".into(),
                error: None,
            },
            ReportView {
                label: "perf".into(),
                provider: "local".into(),
                model: "qwen3".into(),
                round: 1,
                content: String::new(),
                error: Some("HTTP 429".into()),
            },
        ];
        let prompt = merge_prompt("задача", &reports, &config(2));
        assert!(prompt.contains("safety"));
        assert!(prompt.contains("HTTP 429"));
        assert!(prompt.contains("не добавлять пунктов"));
        assert!(prompt.contains("Отчёты 2 независимых агентов"));
        assert!(prompt.contains("волна 2") && prompt.contains("волна 1"));
    }

    #[test]
    fn wave_two_prompt_quotes_own_report_and_peer_digest() {
        let mut two_rounds = config(2);
        two_rounds.rounds = 2;
        let prompt = wave_two_prompt(
            &member("m0"),
            "мой отчёт",
            "### m1 (local/qwen3)\nчужой отчёт",
            &two_rounds,
        );
        assert!(prompt.contains("мой отчёт"));
        assert!(prompt.contains("чужой отчёт"));
        assert!(prompt.contains("волна 2 из 2"));
    }

    #[test]
    fn next_round_prompts_quote_peers_not_the_member_itself() {
        let members = vec![
            (member("m0"), crate::provider::Provider {
                id: "local".into(),
                name: "Local".into(),
                base: "http://127.0.0.1/v1".into(),
                key: String::new(),
                env: String::new(),
                anthropic: false,
                custom: false,
                key_source: "none".into(),
                key_error: None,
                saved: false,
            }),
        ];
        let shared: SharedSession = std::sync::Arc::new(std::sync::Mutex::new(Session::new(
            "test".into(),
            serde_json::from_value(json!({"provider":"local","model":"m"})).unwrap(),
        )));
        let index = {
            let mut state = shared.lock().unwrap();
            let mut message = Message::text("assistant", "");
            message.swarm = vec![
                SwarmReport { label: "m0".into(), provider: "local".into(), model: "m".into(), round: 1, status: MemberStatus::Done, step: 0, content: "своё".into(), error: None },
                SwarmReport { label: "m1".into(), provider: "local".into(), model: "m".into(), round: 1, status: MemberStatus::Done, step: 0, content: "чужое".into(), error: None },
            ];
            state.messages.push(message);
            state.messages.len() - 1
        };
        let prompts = next_round_prompts(&shared, index, &members, &two_rounds_for_tests(), "задача");
        assert_eq!(prompts.len(), 1);
        assert!(prompts[0].contains("своё"));
        assert!(prompts[0].contains("чужое"));
    }
    #[test]
    fn a_later_step_is_a_field_not_part_of_the_status_word() {
        // Step two and beyond used to be spelled `running · шаг N` inside
        // `status`, which forced every reader to compare by prefix. The word now
        // says only the phase and the number rides beside it.
        let shared: SharedSession = std::sync::Arc::new(std::sync::Mutex::new(Session::new(
            "test".into(),
            serde_json::from_value(json!({"provider": "local", "model": "m"})).unwrap(),
        )));
        let index = {
            let mut state = shared.lock().unwrap();
            let mut message = Message::text("assistant", "");
            message.swarm = vec![SwarmReport {
                label: "m0".into(),
                provider: "local".into(),
                model: "m".into(),
                round: 1,
                ..Default::default()
            }];
            state.messages.push(message);
            state.messages.len() - 1
        };
        note(&shared, index, 0, 1, MemberStatus::Running, Some(3));
        let state = shared.lock().unwrap();
        let report = &state.messages[index].swarm[0];
        assert_eq!(report.status, MemberStatus::Running);
        assert_eq!(report.step, 3);
        assert_eq!(serde_json::to_value(report).unwrap()["status"], json!("running"));
        assert_eq!(serde_json::to_value(report).unwrap()["step"], json!(3));
    }

    fn two_rounds_for_tests() -> SwarmConfig {
        let mut c = config(2);
        c.rounds = 2;
        c
    }

    #[test]
    fn failure_notice_is_only_emitted_when_someone_failed() {
        let ok = vec![ReportView {
            label: "a".into(),
            provider: "local".into(),
            model: "m".into(),
            round: 1,
            content: "x".into(),
            error: None,
        }];
        assert!(failure_notice(&ok, 1).is_none());
        let mut bad = ok.clone();
        bad.push(ReportView {
            label: "b".into(),
            provider: "local".into(),
            model: "m".into(),
            round: 1,
            content: String::new(),
            error: Some("boom".into()),
        });
        let notice = failure_notice(&bad, 2).unwrap();
        assert!(notice.contains("1 из 2"));
    }

    #[test]
    fn member_settings_disable_writes_and_json_mode() {
        let mut settings: Settings = serde_json::from_value(json!({
            "provider": "openai",
            "model": "big",
            "mode": "swarm",
            "allow_writes": true,
            "json_mode": true
        }))
        .unwrap();
        settings.swarm = config(2);
        let resolved = member_settings(&settings, &settings.swarm.members[0], &settings.swarm);
        assert!(matches!(resolved.mode, Mode::Chat));
        assert!(!resolved.allow_writes);
        assert!(!resolved.json_mode);
        assert_eq!(resolved.max_steps, settings.swarm.max_steps_per_member);
        let merge = merge_settings(&settings, &settings.swarm);
        assert!(!merge.allow_writes);
        assert!(!merge.json_mode);
    }
}
