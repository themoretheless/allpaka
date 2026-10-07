//! Session-owned, bounded background processes with event-driven waiting.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    sync::{oneshot, watch},
};

const OUTPUT_LIMIT: usize = 128 * 1024;
fn timestamp_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(u64::MAX as u128) as u64
}
struct ProcessGroup(u32);
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.0 as i32), libc::SIGKILL);
        }
    }
}
#[derive(Clone, Default)]
pub(crate) struct Manager(Arc<Mutex<HashMap<String, Arc<Task>>>>, Option<Arc<PathBuf>>);
struct Task {
    storage: Option<Arc<PathBuf>>,
    started: Instant,
    owner: String,
    state: Mutex<Snapshot>,
    cancel: Mutex<Option<oneshot::Sender<()>>>,
    events: watch::Sender<u64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    #[serde(default)]
    name: Option<String>,
    started_ms: u64,
    finished_ms: Option<u64>,
    duration_ms: Option<u64>,
    id: String,
    status: String,
    exit_code: Option<i32>,
    #[serde(default)]
    signal: Option<i32>,
    stdout: String,
    stderr: String,
    stdout_truncated: bool,
    stderr_truncated: bool,
    error: Option<String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record { schema_version:u32, owner:String, snapshot:Snapshot }
impl Task {
    fn persist(&self, phase:&str)->Result<()> {
        if let Some(root)=&self.storage {
            let record=Record{schema_version:1,owner:self.owner.clone(),snapshot:self.state.lock().unwrap().clone()};
            if !std::fs::symlink_metadata(root.as_ref())?.is_dir() {bail!("Invalid background storage");}
            let path=root.join(format!("{}.{}.json",record.snapshot.id,phase));
            let bytes=serde_json::to_vec(&record)?;
            if bytes.len()>2*1024*1024 {bail!("Oversize background receipt");}
            let temp=root.join(format!(".{}.tmp",crate::evaluation::new_id()));
            let result=(||->Result<()> {
                use std::io::Write;
                let mut options=std::fs::OpenOptions::new();options.write(true).create_new(true);
                #[cfg(unix)] {use std::os::unix::fs::OpenOptionsExt;options.mode(0o600);}
                let mut file=options.open(&temp)?;file.write_all(&bytes)?;file.sync_all()?;
                std::fs::hard_link(&temp,&path)?;Ok(())
            })();
            let _=std::fs::remove_file(temp);result?;
        }
        Ok(())
    }
    fn snapshot(&self) -> Value {
        json!(*self.state.lock().unwrap())
    }
    fn changed(&self) {
        self.events.send_modify(|v| *v = v.wrapping_add(1));
    }
    fn terminal(&self) -> bool {
        let state=self.state.lock().unwrap();
        state.status!="running" && (state.status=="interrupted"||self.storage.as_ref().is_none_or(|root|root.join(format!("{}.done.json",state.id)).is_file())||state.error.as_deref()==Some("receipt_storage_error"))
    }
}
impl Manager {
    pub(crate) fn open(data:&Path)->Result<Self> {
        use std::io::Read;
        let root=data.join("background-tasks");std::fs::create_dir_all(&root)?;
        if !std::fs::symlink_metadata(&root)?.is_dir() {bail!("Invalid background storage");}
        let manager=Self(Default::default(),Some(Arc::new(root.clone())));
        let mut records:HashMap<String,Record>=HashMap::new();let mut files=0;
        for entry in std::fs::read_dir(&root)? {
            let entry=entry?;let name=entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".start.json")&&!name.ends_with(".done.json") {continue;}
            files+=1;if files>256||!entry.file_type()?.is_file()||entry.metadata()?.len()>2*1024*1024 {bail!("Invalid background receipt bounds");}
            let mut bytes=Vec::new();std::fs::File::open(entry.path())?.take(2*1024*1024+1).read_to_end(&mut bytes)?;
            if bytes.len()>2*1024*1024 {bail!("Oversize background receipt");}
            let record:Record=serde_json::from_value(crate::strict_json::parse(std::str::from_utf8(&bytes)?)?)?;
            let state=&record.snapshot;
            let phase=if name.ends_with(".done.json"){"done"}else{"start"};
            if record.schema_version!=1||record.owner.is_empty()||record.owner.len()>200||state.id.is_empty()||state.id.len()>100||!state.id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-')||name!=format!("{}.{}.json",state.id,phase)||state.stdout.len()>OUTPUT_LIMIT*3||state.stderr.len()>OUTPUT_LIMIT*3 {bail!("Invalid background receipt identity");}
            if (phase=="start"&&state.status!="running")||(phase=="done"&&!matches!(state.status.as_str(),"completed"|"failed"|"cancelled"|"timed_out")) {bail!("Invalid background receipt status");}
            if state.name.as_ref().is_some_and(|name|name.trim().is_empty()||name.len()>200)||state.signal.is_some_and(|signal|signal<=0||signal>128||state.exit_code.is_some())||state.started_ms==0||state.error.as_ref().is_some_and(|error|error.len()>2000) {bail!("Invalid background receipt metadata");}
            if phase=="start"&&(state.finished_ms.is_some()||state.duration_ms.is_some()||state.exit_code.is_some()||state.signal.is_some()||state.error.is_some()||!state.stdout.is_empty()||!state.stderr.is_empty()||state.stdout_truncated||state.stderr_truncated) {bail!("Invalid background start receipt");}
            if phase=="done"&&(state.finished_ms.is_none()||state.duration_ms.is_none()||(state.status=="completed"&&(state.exit_code!=Some(0)||state.error.is_some()))||(state.status=="failed"&&state.exit_code==Some(0))) {bail!("Invalid background terminal receipt");}
            if let Some(previous)=records.get(&state.id) {if previous.owner!=record.owner||previous.snapshot.started_ms!=state.started_ms||previous.snapshot.name!=state.name {bail!("Background receipt pair differs");}}
            if phase=="done"||!records.contains_key(&state.id){records.insert(state.id.clone(),record);}
        }
        for (_,mut record) in records {
            if record.snapshot.status=="running" {record.snapshot.status="interrupted".into();record.snapshot.exit_code=None;record.snapshot.finished_ms=None;record.snapshot.duration_ms=None;record.snapshot.error=Some("server_restart".into());}
            let (events,_)=watch::channel(0);
            let id=record.snapshot.id.clone();
            manager.0.lock().unwrap().insert(id,Arc::new(Task{storage:manager.1.clone(),started:Instant::now(),owner:record.owner,state:Mutex::new(record.snapshot),cancel:Mutex::new(None),events}));
        }
        Ok(manager)
    }
    fn get(&self, owner: &str, id: &str) -> Result<Arc<Task>> {
        self.0
            .lock()
            .unwrap()
            .get(id)
            .filter(|t| t.owner == owner)
            .cloned()
            .context("Background task not found in this conversation")
    }
    pub(crate) fn cancel_owned(&self, owner: &str) {
        for task in self.0.lock().unwrap().values().filter(|t| t.owner == owner) {
            if let Some(sender) = task.cancel.lock().unwrap().take() {
                let _ = sender.send(());
            }
        }
    }
    pub(crate) fn list(&self, owner: &str) -> Value {
        let mut rows: Vec<_> = self
            .0
            .lock()
            .unwrap()
            .values()
            .filter(|t| t.owner == owner)
            .map(|t| {
                let state=t.state.lock().unwrap();
                json!({"id":state.id,"name":state.name,"status":state.status,"started_ms":state.started_ms,"finished_ms":state.finished_ms,"duration_ms":state.duration_ms,"exit_code":state.exit_code,"signal":state.signal,"stdout_bytes":state.stdout.len(),"stderr_bytes":state.stderr.len(),"stdout_truncated":state.stdout_truncated,"stderr_truncated":state.stderr_truncated,"error":state.error})
            })
            .collect();
        rows.sort_by_key(|v| v["id"].as_str().unwrap_or("").to_owned());
        let mut status_counts=std::collections::BTreeMap::<String,usize>::new();
        let mut known_duration_ms=0u64;let mut unknown_duration_count=0usize;
        for row in &rows {
            *status_counts.entry(row["status"].as_str().unwrap_or("unknown").to_owned()).or_default()+=1;
            match row["duration_ms"].as_u64(){Some(value)=>known_duration_ms=known_duration_ms.saturating_add(value),None=>unknown_duration_count+=1}
        }
        json!({"tasks":rows,"persistent":self.1.is_some(),"automatic_replay":false,"summary":{"status_counts":status_counts,"known_duration_ms":known_duration_ms,"unknown_duration_count":unknown_duration_count,"duration_semantics":"sum_task_elapsed"}})
    }
    pub(crate) async fn execute(&self, owner: &str, cwd: &Path, args: &Value) -> Result<Value> {
        match args["action"].as_str().context("action is required")? {
            "start" => self.start(owner, cwd, args),
            "list" => Ok(self.list(owner)),
            "export" => {
                let include_outputs=match args.get("include_outputs"){None=>false,Some(value)=>value.as_bool().context("include_outputs must be boolean")?};
                let tasks=self.0.lock().unwrap();let mut rows=Vec::new();
                for task in tasks.values().filter(|task|task.owner==owner){
                    let state=task.state.lock().unwrap();
                    let mut row=json!({"id":state.id,"name":state.name,"status":state.status,"started_ms":state.started_ms,"finished_ms":state.finished_ms,"duration_ms":state.duration_ms,"exit_code":state.exit_code,"signal":state.signal,"has_error":state.error.is_some(),"stdout_bytes":state.stdout.len(),"stderr_bytes":state.stderr.len(),"stdout_truncated":state.stdout_truncated,"stderr_truncated":state.stderr_truncated});
                    if include_outputs{row["stdout"]=json!(state.stdout);row["stderr"]=json!(state.stderr);}
                    rows.push(row);
                }
                rows.sort_by(|a,b|a["id"].as_str().cmp(&b["id"].as_str()));
                let packet=json!({"kind":"background_export","schema_version":1,"session_id":owner,"tasks":rows,"outputs_included":include_outputs,"automatic_replay":false});
                if serde_json::to_vec(&packet)?.len()>8*1024*1024 {bail!("Background export exceeds 8 MiB");}
                Ok(packet)
            }
            "status" | "output" => Ok(self
                .get(
                    owner,
                    args["task_id"].as_str().context("task_id is required")?,
                )?
                .snapshot()),
            "cancel" => {
                let task = self.get(
                    owner,
                    args["task_id"].as_str().context("task_id is required")?,
                )?;
                let requested = task
                    .cancel
                    .lock()
                    .unwrap()
                    .take()
                    .is_some_and(|tx| tx.send(()).is_ok());
                Ok(json!({"cancel_requested":requested,"task":task.snapshot()}))
            }
            "cleanup" => {
                let mut tasks = self.0.lock().unwrap();
                let selected=args.get("task_id").map(|value|value.as_str().context("task_id must be a string")).transpose()?;
                if let Some(id)=selected {
                    let task=tasks.get(id).filter(|task|task.owner==owner).context("Background task not found in this conversation")?;
                    if !task.terminal(){bail!("Only terminal background receipts can be removed");}
                }
                let before = tasks.len();
                let removed_ids:std::collections::HashSet<_>=tasks.iter().filter(|(id,t)|t.owner==owner&&t.terminal()&&selected.is_none_or(|selected|selected==id.as_str())).map(|(id,_)|id.clone()).collect();
                if let Some(root)=&self.1 {
                    for id in &removed_ids {
                        for phase in ["done","start"] {let path=root.join(format!("{id}.{phase}.json"));match std::fs::remove_file(path){Ok(())=>{},Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{},Err(e)=>return Err(e.into())}}
                    }
                }
                tasks.retain(|id, _| !removed_ids.contains(id));
                Ok(json!({"removed":before-tasks.len()}))
            }
            "wait" => self.wait(owner, args).await,
            _ => bail!("Unknown background action"),
        }
    }
    fn start(&self, owner: &str, cwd: &Path, args: &Value) -> Result<Value> {
        if let Some(follow_up)=args.get("follow_up") {
            if !follow_up.as_str().is_some_and(|text|!text.trim().is_empty()&&text.len()<=16000) {bail!("follow_up must contain 1-16000 bytes");}
        }
        let name=args.get("name").map(|value|value.as_str().filter(|name|!name.trim().is_empty()&&name.len()<=200).map(str::to_owned).context("name must contain 1-200 bytes")).transpose()?;
        let command = args["command"].as_str().context("command is required")?;
        if command.trim().is_empty() || command.len() > 16_000 {
            bail!("command must contain 1–16000 bytes");
        }
        let timeout = integer(args, "timeout", 3600, 1, 86400)?;
        // Hold registry lock across admission + spawn so concurrent starts cannot exceed limits.
        let mut tasks = self.0.lock().unwrap();
        if tasks.len() >= 128
            || tasks
                .values()
                .filter(|t| t.owner == owner && !t.terminal())
                .count()
                >= 8
        {
            bail!("Background task capacity reached; cancel or clean up tasks");
        }
        #[cfg(unix)]
        let mut process = {
            let mut c = tokio::process::Command::new("/bin/sh");
            c.arg("-c").arg(command);
            c
        };
        #[cfg(windows)]
        let mut process = {
            let mut c = tokio::process::Command::new("cmd.exe");
            c.arg("/C").arg(command);
            c
        };
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            process.as_std_mut().process_group(0);
        }
        process
            .current_dir(cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        let mut child = process
            .spawn()
            .context("Failed to start background command")?;
        let pid = child.id().context("Process ID unavailable")?;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = format!(
            "bg-{}-{}-{}",
            timestamp_ms(),
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let (events, _) = watch::channel(0);
        let task = Arc::new(Task {
            storage: self.1.clone(),
            started: Instant::now(),
            owner: owner.into(),
            state: Mutex::new(Snapshot {
                name,
                started_ms: timestamp_ms(),
                finished_ms: None,
                duration_ms: None,
                id: id.clone(),
                status: "running".into(),
                exit_code: None,
                signal: None,
                stdout: String::new(),
                stderr: String::new(),
                stdout_truncated: false,
                stderr_truncated: false,
                error: None,
            }),
            cancel: Mutex::new(Some(cancel_tx)),
            events,
        });
        let stdout = child.stdout.take().context("Missing stdout pipe")?;
        let stderr = child.stderr.take().context("Missing stderr pipe")?;
        let group = ProcessGroup(pid);
        task.persist("start")?;
        tasks.insert(id, task.clone());
        let output = task.snapshot();
        tokio::spawn(async move {
            let _group = group;
            let out = tokio::spawn(drain(stdout, task.clone(), false));
            let err = tokio::spawn(drain(stderr, task.clone(), true));
            let (reason, result) = tokio::select! {
                result=child.wait()=> (None,result),
                _=cancel_rx => { terminate(&mut child,pid).await; (Some("cancelled"),child.wait().await) },
                _=tokio::time::sleep(Duration::from_secs(timeout)) => { terminate(&mut child,pid).await; (Some("timed_out"),child.wait().await) },
            };
            // A shell can exit while descendants keep its output pipes open. Reap the group.
            #[cfg(unix)]
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
            for mut reader in [out, err] {
                if tokio::time::timeout(Duration::from_secs(2), &mut reader)
                    .await
                    .is_err()
                {
                    reader.abort();
                }
            }
            {
                let mut state = task.state.lock().unwrap();
                state.finished_ms = Some(timestamp_ms());
                state.duration_ms = Some(task.started.elapsed().as_millis().min(u64::MAX as u128) as u64);
                state.status = reason
                    .unwrap_or(if result.as_ref().is_ok_and(|s| s.success()) {
                        "completed"
                    } else {
                        "failed"
                    })
                    .into();
                match result {
                    Ok(s) => {state.exit_code = s.code();
                        #[cfg(unix)] {use std::os::unix::process::ExitStatusExt;state.signal=s.signal();}
                    },
                    Err(e) => state.error = Some(e.to_string()),
                }
            }
            task.cancel.lock().unwrap().take();
            if task.persist("done").is_err() {task.state.lock().unwrap().error=Some("receipt_storage_error".into());}
            task.changed();
        });
        Ok(output)
    }
    async fn wait(&self, owner: &str, args: &Value) -> Result<Value> {
        let ids = args["task_ids"]
            .as_array()
            .context("task_ids is required")?;
        if ids.is_empty() || ids.len() > 32 {
            bail!("wait requires 1–32 task IDs");
        }
        let timeout = integer(args, "wait_seconds", 60, 0, 60)?;
        let tasks: Vec<_> = ids
            .iter()
            .map(|v| self.get(owner, v.as_str().context("task IDs must be strings")?))
            .collect::<Result<_>>()?;
        let receivers: Vec<_> = tasks.iter().map(|t| t.events.subscribe()).collect();
        let terminal = || tasks.iter().any(|t| t.terminal());
        let mut expired = false;
        if !terminal() && timeout > 0 {
            let futures: Vec<_> = receivers
                .into_iter()
                .zip(tasks.iter().cloned())
                .map(|(mut rx, task)| {
                    Box::pin(async move {
                        loop {
                            if task.terminal() {
                                break;
                            }
                            if rx.changed().await.is_err() {
                                break;
                            }
                        }
                    })
                })
                .collect();
            expired = tokio::time::timeout(
                Duration::from_secs(timeout),
                futures_util::future::select_all(futures),
            )
            .await
            .is_err();
        } else if !terminal() {
            expired = true;
        }
        Ok(
            json!({"tasks":tasks.iter().map(|t|t.snapshot()).collect::<Vec<_>>(),"timed_out":expired}),
        )
    }
}
fn integer(args: &Value, key: &str, default: u64, min: u64, max: u64) -> Result<u64> {
    let n = args
        .get(key)
        .map(|v| {
            v.as_u64()
                .with_context(|| format!("{key} must be an integer"))
        })
        .transpose()?
        .unwrap_or(default);
    if n < min || n > max {
        bail!("{key} must be {min}–{max}");
    }
    Ok(n)
}
async fn terminate(child: &mut tokio::process::Child, pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let _ = tokio::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output()
            .await;
    }
    let _ = child.start_kill();
}
async fn drain(mut stream: impl AsyncRead + Unpin, task: Arc<Task>, stderr: bool) {
    let mut retained = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = match stream.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                task.state.lock().unwrap().error = Some(format!("Output read failed: {e}"));
                break;
            }
        };
        let take = n.min(OUTPUT_LIMIT - retained.len());
        if take == 0 {
            let mut s = task.state.lock().unwrap();
            if stderr {
                s.stderr_truncated = true;
            } else {
                s.stdout_truncated = true;
            }
            continue;
        }
        retained.extend_from_slice(&buf[..take]);
        {
            let mut s = task.state.lock().unwrap();
            let mut text = String::from_utf8_lossy(&retained).into_owned();
            if text.len() > OUTPUT_LIMIT {
                let mut end = OUTPUT_LIMIT;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                if stderr {
                    s.stderr_truncated = true;
                } else {
                    s.stdout_truncated = true;
                }
            }
            if stderr {
                s.stderr = text;
                s.stderr_truncated |= take < n;
            } else {
                s.stdout = text;
                s.stdout_truncated |= take < n;
            }
        }
        task.changed();
    }
}
pub(crate) fn schema() -> Value {
    json!({"type":"function","function":{"name":"background","description":"Manage session-owned background commands: start, list, status, output, wait, cancel, cleanup. Prefer event-driven wait to polling. Output capped at 128 KiB per stream. Completed output receipts survive server restart; unfinished receipts recover as interrupted without replay. wait returns when any requested task completes, or after at most 60 seconds.","parameters":{"type":"object","properties":{"include_outputs":{"type":"boolean","description":"Include intentionally captured command output in export only when explicitly requested."},"action":{"type":"string","enum":["start","list","status","output","wait","cancel","cleanup","export"]},"command":{"type":"string"},"name":{"type":"string","description":"Optional human-readable task label, at most 200 UTF-8 bytes."},"follow_up":{"type":"string","description":"Explicit task continuation after termination. Does not override Stop or survive restart."},"task_id":{"type":"string"},"task_ids":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":32},"timeout":{"type":"integer","minimum":1,"maximum":86400},"wait_seconds":{"type":"integer","minimum":0,"maximum":60}},"required":["action"],"additionalProperties":false}}})
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    async fn finish(manager: &Manager, owner: &str, id: &str) -> Value {
        manager
            .execute(
                owner,
                Path::new("/tmp"),
                &json!({"action":"wait","task_ids":[id],"wait_seconds":5}),
            )
            .await
            .unwrap()["tasks"][0]
            .clone()
    }
    #[tokio::test]
    async fn persistent_receipts_recover_finished_output_and_interrupt_orphans_without_replay() {
        let directory=std::env::temp_dir().join(format!("allpaka-bg-{}",crate::evaluation::new_id()));
        let manager=Manager::open(&directory).unwrap();
        let running=manager.execute("owner",Path::new("/tmp"),&json!({"action":"start","command":"printf retained","name":"Сборка"})).await.unwrap();
        let id=running["id"].as_str().unwrap();
        let done=finish(&manager,"owner",id).await;
        assert_eq!(done["stdout"],"retained");assert_eq!(done["name"],"Сборка");
        {use std::os::unix::fs::PermissionsExt;assert_eq!(std::fs::metadata(directory.join(format!("background-tasks/{id}.done.json"))).unwrap().permissions().mode()&0o777,0o600);}

        let receipt_path=directory.join(format!("background-tasks/{id}.done.json"));
        let original=std::fs::read(&receipt_path).unwrap();
        let saved:Value=serde_json::from_slice(&original).unwrap();
        for (field,value) in [("owner",json!("different")),("started_ms",json!(0)),("duration_ms",Value::Null),("exit_code",json!(7)),("name",json!("x".repeat(201)))] {
            let mut corrupt=saved.clone();if field=="owner" {corrupt[field]=value;}else{corrupt["snapshot"][field]=value;}
            std::fs::write(&receipt_path,serde_json::to_vec(&corrupt).unwrap()).unwrap();
            assert!(Manager::open(&directory).is_err());
        }
        std::fs::write(&receipt_path,&original).unwrap();
        let start_path=directory.join(format!("background-tasks/{id}.start.json"));
        let start_original=std::fs::read(&start_path).unwrap();
        for path in [&start_path,&receipt_path]{let mut legacy:Value=serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();legacy["snapshot"].as_object_mut().unwrap().remove("name");legacy["snapshot"].as_object_mut().unwrap().remove("signal");std::fs::write(path,serde_json::to_vec(&legacy).unwrap()).unwrap();}
        let legacy=Manager::open(&directory).unwrap().get("owner",id).unwrap().snapshot();assert!(legacy["name"].is_null());assert_eq!(legacy["stdout"],"retained");
        std::fs::write(&start_path,start_original).unwrap();std::fs::write(&receipt_path,&original).unwrap();
        let mut orphan=manager.get("owner",id).unwrap().state.lock().unwrap().clone();
        orphan.id="bg-orphan".into();orphan.status="running".into();orphan.stdout.clear();orphan.finished_ms=None;orphan.duration_ms=None;orphan.exit_code=None;
        crate::evaluation::commit_new(&directory.join("background-tasks/bg-orphan.start.json"),&Record{schema_version:1,owner:"owner".into(),snapshot:orphan}).unwrap();
        let restored=Manager::open(&directory).unwrap();
        assert_eq!(restored.get("owner",id).unwrap().snapshot(),done);
        let interrupted=restored.get("owner","bg-orphan").unwrap().snapshot();
        assert_eq!(interrupted["status"],"interrupted");assert!(interrupted["duration_ms"].is_null());assert!(interrupted["exit_code"].is_null());
        assert!(restored.get("other",id).is_err());
        let metadata=restored.execute("owner",Path::new("/tmp"),&json!({"action":"export"})).await.unwrap();assert_eq!(metadata["tasks"].as_array().unwrap().len(),2);assert_eq!(metadata["outputs_included"],false);assert!(!serde_json::to_string(&metadata).unwrap().contains("retained"));
        let outputs=restored.execute("owner",Path::new("/tmp"),&json!({"action":"export","include_outputs":true})).await.unwrap();assert!(outputs["tasks"].as_array().unwrap().iter().any(|row|row["stdout"]=="retained"));
        assert_eq!(restored.execute("other",Path::new("/tmp"),&json!({"action":"export"})).await.unwrap()["tasks"],json!([]));
        assert!(restored.execute("owner",Path::new("/tmp"),&json!({"action":"export","include_outputs":"true"})).await.is_err());

        assert_eq!(restored.list("owner")["persistent"],true);
        assert!(restored.execute("other",Path::new("/tmp"),&json!({"action":"cleanup","task_id":id})).await.is_err());
        assert_eq!(restored.execute("owner",Path::new("/tmp"),&json!({"action":"cleanup","task_id":id})).await.unwrap()["removed"],1);
        assert!(restored.get("owner",id).is_err());assert!(restored.get("owner","bg-orphan").is_ok());
        let reopened=Manager::open(&directory).unwrap();assert!(reopened.get("owner",id).is_err());assert!(reopened.get("owner","bg-orphan").is_ok());
        restored.execute("owner",Path::new("/tmp"),&json!({"action":"cleanup"})).await.unwrap();
        assert!(restored.list("owner")["tasks"].as_array().unwrap().is_empty());
        assert!(Manager::open(&directory).unwrap().list("owner")["tasks"].as_array().unwrap().is_empty());
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[tokio::test]
    async fn unix_signal_termination_is_distinct_from_exit_code() {
        let directory=std::env::temp_dir().join(format!("allpaka-signal-{}",crate::evaluation::new_id()));
        let manager=Manager::open(&directory).unwrap();let running=manager.execute("signal",Path::new("/tmp"),&json!({"action":"start","command":"kill -TERM $$"})).await.unwrap();
        let done=finish(&manager,"signal",running["id"].as_str().unwrap()).await;
        assert_eq!(done["status"],"failed");assert!(done["exit_code"].is_null());assert_eq!(done["signal"],15);
        assert_eq!(manager.list("signal")["tasks"][0]["signal"],15);
        assert_eq!(Manager::open(&directory).unwrap().get("signal",running["id"].as_str().unwrap()).unwrap().snapshot(),done);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[tokio::test]
    async fn timing_is_unknown_until_terminal_and_available_in_catalog() {
        let manager=Manager::default();
        let running=manager.execute("timing",Path::new("/tmp"),&json!({"action":"start","command":"sleep 0.05"})).await.unwrap();
        assert!(manager.execute("timing",Path::new("/tmp"),&json!({"action":"cleanup","task_id":running["id"]})).await.is_err());
        assert!(running["started_ms"].as_u64().unwrap()>0);
        assert!(running["finished_ms"].is_null());
        assert!(running["duration_ms"].is_null());
        let live=manager.list("timing");assert_eq!(live["summary"]["status_counts"]["running"],1);assert_eq!(live["summary"]["known_duration_ms"],0);assert_eq!(live["summary"]["unknown_duration_count"],1);
        let done=finish(&manager,"timing",running["id"].as_str().unwrap()).await;
        assert_eq!(done["status"],"completed");
        assert!(done["finished_ms"].as_u64().is_some());
        assert!(done["duration_ms"].as_u64().unwrap()>=40);
        let catalog=manager.list("timing");assert_eq!(catalog["tasks"][0]["duration_ms"],done["duration_ms"]);assert_eq!(catalog["summary"]["status_counts"]["completed"],1);assert_eq!(catalog["summary"]["unknown_duration_count"],0);assert_eq!(catalog["summary"]["known_duration_ms"],done["duration_ms"]);
        assert_eq!(manager.list("other")["summary"]["known_duration_ms"],0);
    }
    #[tokio::test]
    async fn captures_output_and_exit_code_and_enforces_ownership() {
        let manager = Manager::default();
        let task = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":"printf 'привет'; printf error >&2; exit 7"}),
            )
            .await
            .unwrap();
        let id = task["id"].as_str().unwrap();
        assert!(manager
            .execute(
                "b",
                Path::new("/tmp"),
                &json!({"action":"cancel","task_id":id})
            )
            .await
            .is_err());
        let done = finish(&manager, "a", id).await;
        assert_eq!(done["status"], "failed");
        assert_eq!(done["exit_code"], 7);
        assert_eq!(done["stdout"], "привет");
        assert_eq!(done["stderr"], "error");
        assert_eq!(manager.list("b")["tasks"], json!([]));
        assert_eq!(
            manager
                .execute("a", Path::new("/tmp"), &json!({"action":"cleanup"}))
                .await
                .unwrap()["removed"],
            1
        );
    }
    #[tokio::test]
    async fn cancellation_and_timeout_stop_descendant_processes() {
        let manager = Manager::default();
        let task = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":"sleep 30 & echo $!; wait"}),
            )
            .await
            .unwrap();
        let id = task["id"].as_str().unwrap();
        let child = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let value = manager.get("a", id).unwrap().snapshot();
                if let Ok(pid) = value["stdout"].as_str().unwrap().trim().parse::<i32>() {
                    break pid;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"cancel","task_id":id}),
            )
            .await
            .unwrap();
        assert_eq!(finish(&manager, "a", id).await["status"], "cancelled");
        // kill(0) can see a transient zombie; ps distinguishes a stopped process.
        let output = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &child.to_string()])
            .output()
            .unwrap();
        let state = String::from_utf8_lossy(&output.stdout);
        assert!(
            state.trim().is_empty() || state.trim().starts_with('Z'),
            "descendant still running: {state}"
        );
        let task = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":"sleep 30","timeout":1}),
            )
            .await
            .unwrap();
        assert_eq!(
            finish(&manager, "a", task["id"].as_str().unwrap()).await["status"],
            "timed_out"
        );
    }
    #[tokio::test]
    async fn waits_for_any_completion_and_limits_output() {
        let manager = Manager::default();
        let slow = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":"sleep 30"}),
            )
            .await
            .unwrap();
        let fast = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":"yes x | head -c 200000"}),
            )
            .await
            .unwrap();
        let result = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"wait","task_ids":[slow["id"],fast["id"]],"wait_seconds":5}),
            )
            .await
            .unwrap();
        assert_eq!(result["timed_out"], false);
        assert_eq!(result["tasks"][0]["status"], "running");
        assert_eq!(result["tasks"][1]["status"], "completed");
        assert_eq!(
            result["tasks"][1]["stdout"].as_str().unwrap().len(),
            OUTPUT_LIMIT
        );
        assert_eq!(result["tasks"][1]["stdout_truncated"], true);
        manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"cancel","task_id":slow["id"]}),
            )
            .await
            .unwrap();
        finish(&manager, "a", slow["id"].as_str().unwrap()).await;
    }
    #[tokio::test]
    async fn admission_limit_and_invalid_utf8_are_bounded() {
        let manager = Manager::default();
        let mut ids = Vec::new();
        for _ in 0..8 {
            let task = manager
                .execute(
                    "a",
                    Path::new("/tmp"),
                    &json!({"action":"start","command":"sleep 30"}),
                )
                .await
                .unwrap();
            ids.push(task["id"].as_str().unwrap().to_owned());
        }
        assert!(manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":"true"})
            )
            .await
            .is_err());
        manager.cancel_owned("a");
        for id in ids {
            assert_eq!(finish(&manager, "a", &id).await["status"], "cancelled");
        }
        let task = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":r"head -c 200000 /dev/zero | tr '\000' '\377'"}),
            )
            .await
            .unwrap();
        let done = finish(&manager, "a", task["id"].as_str().unwrap()).await;
        assert_eq!(done["status"], "completed");
        assert!(done["stdout"].as_str().unwrap().len() <= OUTPUT_LIMIT);
        assert_eq!(done["stdout_truncated"], true);
    }
    #[tokio::test]
    async fn wait_timeout_leaves_the_process_running() {
        let manager = Manager::default();
        let task = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"start","command":"sleep 30"}),
            )
            .await
            .unwrap();
        let id = task["id"].as_str().unwrap();
        let result = manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"wait","task_ids":[id],"wait_seconds":0}),
            )
            .await
            .unwrap();
        assert_eq!(result["timed_out"], true);
        assert_eq!(result["tasks"][0]["status"], "running");
        assert!(manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"wait","task_ids":[id],"wait_seconds":61})
            )
            .await
            .is_err());
        manager
            .execute(
                "a",
                Path::new("/tmp"),
                &json!({"action":"cancel","task_id":id}),
            )
            .await
            .unwrap();
        finish(&manager, "a", id).await;
    }
}
