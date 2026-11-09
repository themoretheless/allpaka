//! The per-session actor: queued actions run one at a time, with
//! cancellation, and the session is persisted after each.

use crate::*;

pub(crate) fn save(app: &App, s: &Session) -> Result<()> {
    if s.ephemeral.is_some() {
        return Ok(());
    }
    let file = app.data.join(format!("{}.json", s.id));
    let temp = app.data.join(format!("{}.tmp", s.id));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    use std::io::Write;
    let mut f = options.open(&temp)?;
    f.write_all(&serde_json::to_vec(s)?)?;
    f.sync_all()?;
    std::fs::rename(temp, file)?;
    #[cfg(unix)]
    std::fs::File::open(app.data.as_ref())?.sync_all()?;
    Ok(())
}
pub(crate) fn persist(app: &App, s: &SharedSession) {
    let mut s = s.lock().unwrap();
    if let Err(e) = save(app, &s) {
        s.error = Some(format!("History could not be saved: {e}"));
    }
}
#[derive(Debug)]
pub(crate) enum TurnOutcome {
    Complete,
    NoProgress,
    GoalIncomplete,
    TokenLimit,
    Compacted,
}

pub(crate) async fn cancel(task: &mut Option<JoinHandle<Result<TurnOutcome>>>, s: &SharedSession) {
    if let Some(t) = task.take() {
        t.abort();
        let _ = t.await;
    }
    let mut s = s.lock().unwrap();
    s.background_continuation_epoch = s.background_continuation_epoch.wrapping_add(1);
    s.close_pending_tools();
    // A stopped swarm wave leaves reports that will never finish; say so instead
    // of showing a member that looks like it is still thinking.
    for message in s.messages.iter_mut().rev() {
        if message.swarm.is_empty() {
            continue;
        }
        for report in &mut message.swarm {
            if report.status.unfinished() {
                report.status = MemberStatus::Cancelled;
                if report.error.is_none() {
                    report.error = Some("Отменено пользователем".into());
                }
            }
        }
        break;
    }
    s.status = SessionStatus::Paused;
    if s.notice.as_deref() == Some("Сжатие контекста…") {
        s.notice = None;
    }
}
pub(crate) async fn actor(app: App, s: SharedSession, mut rx: mpsc::Receiver<Action>) {
    let mut task: Option<JoinHandle<Result<TurnOutcome>>> = None;
    let mut paused = true;
    loop {
        if !paused && task.is_none() {
            let pending = {
                let mut s = s.lock().unwrap();
                if s.folder != HistoryFolder::Active {
                    paused = true;
                    None
                } else if s.queue.is_empty() {
                    None
                } else {
                    Some(s.queue.remove(0))
                }
            };
            if let Some(p) = pending {
                {
                    let mut s = s.lock().unwrap();
                    s.settings = p.settings.clone();
                    s.status = SessionStatus::Running;
                    s.error = None;
                    s.notice = None;
                    s.step = 0;
                    if s.messages.is_empty() {
                        s.title = p.text.chars().take(60).collect();
                    }
                    let mut message = Message::text("user", p.text);
                    message.images = p.images;
                    s.messages.push(message);
                    if s.settings.mode == Mode::Goal && !p.continuation {
                        let origin = s.messages.len() - 1;
                        if let Err(error) = plan::start_goal(&mut s, origin) {
                            s.status = SessionStatus::Error;
                            s.error = Some(error.to_string());
                        }
                    }
                }
                persist(&app, &s);
                if s.lock().unwrap().status == SessionStatus::Error {
                    paused = true;
                    continue;
                }
                task = Some(tokio::spawn(turn(app.clone(), s.clone(), None)));
            }
        }
        tokio::select! {
            command=rx.recv() => {
                let Some(mut a)=command else { cancel(&mut task,&s).await;break; };
                let reply=a.reply.take();
                let mut rejection=None;
                let mut retry_member=None;
                match a.kind {
                    ActionKind::Compact => {
                        if task.is_some() { rejection=Some("Stop generation before manual compaction".into()); }
                        else {
                            {let mut state=s.lock().unwrap();if let Some(settings)=a.settings {state.settings=settings;}state.status=SessionStatus::Running;state.error=None;state.notice=Some("Сжатие контекста…".into());}
                            paused=true;
                            let app=app.clone();let s=s.clone();
                            task=Some(tokio::spawn(async move {compact::run(&app,&s,true,None).await?;Ok(TurnOutcome::Compacted)}));
                        }
                    },
                    ActionKind::RetryMember => {
                        if task.is_some() { rejection=Some("Дождитесь окончания хода или нажмите «Стоп»".into()); }
                        else {
                            let mut state=s.lock().unwrap();
                            if state.settings.mode!=Mode::Swarm { rejection=Some("Повтор участника доступен, пока режим разговора — Swarm".into()); }
                            else { state.status=SessionStatus::Running;state.error=None;retry_member=Some(a.text.trim().to_string()); }
                        }
                    },
                    ActionKind::BackgroundDone => {
                        let mut state=s.lock().unwrap();
                        if a.wake_epoch==Some(state.background_continuation_epoch) && !paused && state.folder==HistoryFolder::Active && matches!(state.settings.mode,Mode::Auto|Mode::Goal) && state.queue.len()<32 {
                            let settings=state.settings.clone();state.queue.push(Pending{continuation:true,images:vec![],text:a.text,settings});
                        }
                    },
                    ActionKind::Stop => { cancel(&mut task,&s).await;paused=true; },
                    ActionKind::Resume => {
                        paused=false;
                        let mut s=s.lock().unwrap();
                        if task.is_none() {
                            let truncated=s.messages.last().is_some_and(|m|m.truncated);
                            if truncated || s.queue.is_empty() {
                                let settings=a.settings.unwrap_or_else(||s.settings.clone());
                                let text=if truncated {"The previous response reached its output token limit. Continue the unfinished answer without repeating existing text. Any incomplete tool calls were NOT executed: regenerate the necessary calls in full, never continue partial JSON. Do not repeat previously completed side effects."} else {"Continue the task from the current state. Do not repeat completed side effects."};
                                s.queue.insert(0,Pending{continuation:true,images:vec![],text:text.into(),settings});
                            }
                        }
                    },
                    ActionKind::ClearQueue => { let mut s=s.lock().unwrap();s.queue.clear();s.steering.clear(); },
                    ActionKind::Steer if task.is_some() => { let mut s=s.lock().unwrap();if s.steering.len()<32 {s.steering.push(a.text);}else{rejection=Some("Steering queue is full".to_string());s.error=rejection.clone();} },
                    ActionKind::Send | ActionKind::SendNow | ActionKind::Steer => {
                        if a.kind==ActionKind::SendNow {cancel(&mut task,&s).await;}
                        let mut s=s.lock().unwrap();
                        let settings=a.settings.unwrap_or_else(||s.settings.clone());
                        let p=Pending{continuation:a.kind==ActionKind::Steer,images:a.images,text:a.text,settings};
                        if s.queue.len()>=32 && a.kind!=ActionKind::SendNow {rejection=Some("Message queue is full".to_string());s.error=rejection.clone();} else {
                            if a.kind==ActionKind::SendNow {s.queue.insert(0,p);} else {s.queue.push(p);}
                            paused=false;
                        }
                    },
                    // answered by the handler before anything is queued
                    ActionKind::Rename | ActionKind::Move => {}
                }
                persist(&app,&s);
                if let Some(reply)=reply {let _=reply.send(rejection.map_or(Ok(()),Err));}
                if let Some(label)=retry_member { task=Some(tokio::spawn(turn(app.clone(), s.clone(), Some(label)))); }
            },
            result=async { task.as_mut().unwrap().await },if task.is_some()=>{
                task=None;
                { let mut s=s.lock().unwrap(); match result {
                    Ok(Ok(TurnOutcome::Compacted))=>{s.status=if s.queue.is_empty(){SessionStatus::Idle}else{SessionStatus::Paused};paused=true;},
                    Ok(Ok(TurnOutcome::TokenLimit))=>{
                        s.status=SessionStatus::Paused;paused=true;
                        s.notice=Some(format!("Достигнут лимит {} токенов. Частичный ответ сохранён; незавершённые инструменты не выполнялись. Можно увеличить лимит ответа и нажать «Продолжить».",s.settings.max_output_tokens));
                    },
                    Ok(Ok(TurnOutcome::NoProgress))=>{s.status=SessionStatus::Paused;paused=true;s.notice=Some("Три одинаковых вызова инструмента подряд завершились одной и той же ошибкой. История сохранена. Уточните задачу или нажмите «Продолжить», чтобы повторить попытку.".into());},
                    Ok(Ok(TurnOutcome::GoalIncomplete))=>{s.status=SessionStatus::Paused;paused=true;s.notice=Some("Цель не завершена: в плане остались незакрытые этапы или отсутствуют критерии и заявленные подтверждения. Прогресс сохранён. Уточните задачу или нажмите «Продолжить».".into());},
                    Ok(Ok(TurnOutcome::Complete))=>{
                        s.status=SessionStatus::Idle;
                        if !s.steering.is_empty(){let settings=s.settings.clone();s.queue.insert(0,Pending{continuation:true,images:vec![],text:"Apply the pending user steering to the current task.".into(),settings});}
                    },
                    Ok(Err(e))=>{s.notice=None;s.status=SessionStatus::Error;s.error=Some(format!("{e:#}"));s.close_pending_tools();paused=true;},
                    Err(e)=>{s.notice=None;s.status=SessionStatus::Error;s.error=Some(format!("Task failed: {e}"));s.close_pending_tools();paused=true;},
                } }
                persist(&app,&s);
            }
        }
    }
}
