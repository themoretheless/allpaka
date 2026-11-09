//! Live trace handles: a trace being recorded, and the span guard that
//! finishes or fails its span when dropped.

use super::*;

#[derive(Clone)]
pub(crate) struct Trace {
    pub(super) record: Arc<Mutex<Record>>,
    pub(super) path: Arc<PathBuf>,
    pub(super) writer: Arc<Mutex<()>>,
    pub(super) cache: TraceCache,
}
pub(crate) struct Guard {
    pub(super) trace: Trace,
    pub(super) index: usize,
    pub(super) started: Instant,
    pub(super) finished: bool,
}

impl Trace {
    pub(crate) fn id(&self) -> String {
        self.record.lock().unwrap().id.clone()
    }
    pub(crate) async fn generation(
        &self,
        name: &str,
        provider_id: &str,
        parent: usize,
        future: impl std::future::Future<Output = Result<(crate::types::Message, Value)>>,
    ) -> Result<(crate::types::Message, Value)> {
        let mut span = self.span("model", name, Some(parent));
        span.set_provider(provider_id);
        let result = future.await;
        match &result {
            Ok((message, usage)) => span.finish(
                if message.truncated {
                    "token_limit"
                } else {
                    "completed"
                },
                usage,
            ),
            Err(_) => span.finish("failed", &Value::Null),
        }
        result
    }
    pub(crate) fn span(&self, kind: &str, name: &str, parent: Option<usize>) -> Guard {
        let index = {
            let mut record = self.record.lock().unwrap();
            let index = record.spans.len();
            record.spans.push(Span {
                id: index,
                parent_id: parent,
                kind: kind.into(),
                name: name.chars().take(200).collect(),
                status: "running".into(),
                started_ms: now(),
                duration_ms: None,
                usage: Value::Null,
                linked_trace_id: None,
                provider_id: None,
            });
            index
        };
        self.flush();
        Guard {
            trace: self.clone(),
            index,
            started: Instant::now(),
            finished: false,
        }
    }
    pub(super) fn persist(&self) -> Result<()> {
        let _writer = self.writer.lock().unwrap();
        let record = self.record.lock().unwrap().clone();
        atomic_write(self.path.as_ref(), &record)?;
        update_cache(&self.cache, self.path.as_ref(), &record);
        Ok(())
    }
    pub(super) fn flush(&self) {
        if let Err(error) = self.persist() {
            eprintln!("Studio trace persistence failed: {error}");
        }
    }
}
impl Guard {
    pub(crate) fn set_provider(&mut self, provider: &str) {
        if provider.is_empty()
            || provider.len() > 100
            || !provider
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-:/".contains(&b))
        {
            return;
        }
        self.trace.record.lock().unwrap().spans[self.index].provider_id = Some(provider.into());
        self.trace.flush();
    }
    pub(crate) fn id(&self) -> usize {
        self.index
    }
    pub(crate) fn link_trace(&mut self, id: &str) {
        if id.is_empty()
            || id.len() > 80
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return;
        }
        self.trace.record.lock().unwrap().spans[self.index].linked_trace_id = Some(id.into());
        self.trace.flush();
    }
    pub(crate) fn finish(&mut self, status: &str, usage: &Value) {
        if self.finished {
            return;
        }
        self.finished = true;
        {
            let mut record = self.trace.record.lock().unwrap();
            let span = &mut record.spans[self.index];
            span.status = status.into();
            span.duration_ms =
                Some(self.started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            span.usage = public_usage(usage);
            if self.index == 0 {
                record.status = status.into();
            }
        }
        self.trace.flush();
        if self.index == 0 && status == "completed" {
            let (project, id) = {
                let record = self.trace.record.lock().unwrap();
                (record.project_id.clone(), record.id.clone())
            };
            if let Some(data) = self
                .trace
                .path
                .parent()
                .and_then(|path| path.parent())
                .and_then(|path| path.parent())
            {
                if let Err(error) = crate::online_evaluation::enqueue_completed(data, &project, &id)
                {
                    eprintln!("Online evaluation admission failed: {error}");
                }
            }
        }
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        if !self.finished {
            self.finish("interrupted", &Value::Null);
        }
    }
}
