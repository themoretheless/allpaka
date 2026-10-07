"""Dependency-free client for Allpaka Studio evaluations and metadata tracing (Python 3.9+)."""
import json
import hashlib
import csv
import io
import math
import time
import asyncio
import contextvars
import threading
import functools
import inspect
from contextlib import nullcontext
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_ACTIVE_TRACE = contextvars.ContextVar('allpaka_active_trace', default=None)


class EvaluationError(Exception):
    def __init__(self, reason, run_id=None, cancellation_failed=False):
        super().__init__(reason)
        self.reason = reason
        self.run_id = run_id
        self.cancellation_failed = cancellation_failed


def _reject_nonfinite(_):
    raise ValueError("Nonfinite JSON number")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _TraceSpan:
    def __init__(self, trace, name, kind, provider_id=None):
        self.trace, self.name, self.kind = trace, name, kind
        self.provider_id = provider_id
        self.index = None

    def __enter__(self):
        trace = self.trace
        with trace._lock:
            if not hasattr(trace, '_start_ns') or trace._closed or self.index is not None or len(trace.spans) >= 200:
                raise ValueError('Trace is closed, span reused or span limit reached')
            parent = trace._parent.get()
            if parent is not None and trace.spans[parent]['status'] != 'running':
                raise ValueError('Parent span already ended')
            if parent is None and trace.spans:
                raise ValueError('Enter the trace context before creating spans')
            self.index = len(trace.spans)
            self.start = trace._elapsed()
            trace.spans.append(dict(parent_id=parent, kind=self.kind, name=self.name,
                started_ms=trace.started_ms+self.start, duration_ms=None, status='running', usage={}))
            if self.provider_id is not None:
                trace.spans[self.index]['provider_id'] = self.provider_id
        self.token = trace._parent.set(self.index)
        return self

    def set_guardrail_receipt(self, receipt):
        from allpaka_guardrails import _trace_receipt
        frozen=_trace_receipt(receipt)
        expected='guardrail.%s.%s.%s.%s' % (frozen['stage'],frozen['action'],'pass' if frozen['passed'] else 'fail',frozen['policy_sha256'])
        if self.kind!='tool' or self.name!=expected:
            raise ValueError('Guardrail receipt requires matching tool metadata')
        with self.trace._lock:
            if self.index is None or self.trace._closed or self.trace.spans[self.index]['status']!='running':
                raise ValueError('Guardrail receipt requires an active span')
            self.trace.spans[self.index]['usage']['guardrail_receipt']=frozen

    def set_evaluator_ref(self,evaluator_id,version):
        if not isinstance(evaluator_id,str) or not 1<=len(evaluator_id)<=100 or any(not (c.isascii() and (c.isalnum() or c in '._-')) for c in evaluator_id) or type(version) is not int or not 1<=version<=2**64-1:raise ValueError('Use technical evaluator ID and positive version')
        with self.trace._lock:
            if self.index is None or self.trace._closed or self.kind!='tool' or self.trace.spans[self.index]['status']!='running':raise ValueError('Evaluator identity requires active tool span')
            self.trace.spans[self.index]['usage']['evaluator_ref']=dict(id=evaluator_id,version=version)

    def set_evaluation_scores(self,scores):
        if not isinstance(scores,dict) or not 1<=len(scores)<=20:raise ValueError('Use 1-20 evaluation scores')
        frozen=dict(scores)
        for metric,value in frozen.items():
            if not isinstance(metric,str) or not 1<=len(metric)<=100 or any(not (c.isascii() and (c.isalnum() or c in '._-')) for c in metric) or type(value) not in (int,float) or not 0<=value<=1 or not math.isfinite(value):raise ValueError('Use technical metric names and finite scores 0-1')
        with self.trace._lock:
            if self.index is None or self.trace._closed or self.kind!='tool' or self.trace.spans[self.index]['status']!='running':raise ValueError('Scores require an active evaluation tool span')
            self.trace.spans[self.index]['usage']['evaluation_scores']=frozen

    def set_usage(self, **usage):
        for key, value in usage.items():
            if key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'first_text_ms'):
                valid = type(value) is int and 0 <= value <= 2**64-1
            elif key == 'cost':
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            elif key == 'cost_currency':
                valid = isinstance(value, str) and len(value) == 3 and all('A' <= c <= 'Z' for c in value)
            else:
                valid = False
            if not valid:
                raise ValueError('Unsupported or invalid trace usage field')
        with self.trace._lock:
            if self.index is None or self.trace._closed or self.trace.spans[self.index]['status'] != 'running':
                raise ValueError('Usage requires an active span')
            self.trace.spans[self.index]['usage'].update(usage)

    def __exit__(self, error_type, error, traceback):
        self.trace._parent.reset(self.token)
        return self._finish(error_type,error)

    def _finish(self, error_type=None, error=None):
        if error_type is None and getattr(self,'_requires_terminal',False):
            if not getattr(self,'_terminal_seen',False):
                error_type=asyncio.CancelledError;error=asyncio.CancelledError()
            elif getattr(self,'_provider_failure',False):error_type=RuntimeError
        with self.trace._lock:
            if self.trace._closed:
                return False
            record = self.trace.spans[self.index]
            if record['status'] != 'running':
                return False
            end = self.trace._elapsed()
            record['duration_ms'] = end-self.start
            record['status'] = ('interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError))
                                else 'failed' if error_type else 'completed')
            for index, child in enumerate(self.trace.spans):
                parent = child['parent_id']
                while parent is not None and parent != self.index:
                    parent = self.trace.spans[parent]['parent_id']
                if index != self.index and parent == self.index and child['status'] == 'running':
                    child['status'] = 'interrupted'
                    child['duration_ms'] = end-(child['started_ms']-self.trace.started_ms)
                    record['status'] = 'interrupted'
                    self.trace._unfinished = True
        return False


class _TrackedChatStream:
    """Single-consumer iterator; no chunk accumulation or content inspection."""
    def __init__(self, stream, span, usage):
        self._stream=stream;self._iterator=iter(stream);self._span=span;self._usage=usage
        self._closed=False

    def __iter__(self):return self

    def __next__(self):
        if self._closed:raise StopIteration
        token=self._span.trace._parent.set(self._span.index)
        try:
            chunk=next(self._iterator)
            if not self._span.trace._closed and self._span.trace.spans[self._span.index]['status']=='running':
                self._usage(self._span,chunk)
            return chunk
        except StopIteration:
            self._closed=True
            try:self._stream.close()
            except BaseException as error:
                self._span._finish(type(error),error)
                raise
            self._span._finish()
            raise
        except BaseException as error:
            self._closed=True
            try:self._stream.close()
            except BaseException:pass
            self._span._finish(type(error),error)
            raise
        finally:self._span.trace._parent.reset(token)

    def close(self):
        if self._closed:return
        self._closed=True
        try:self._stream.close()
        finally:self._span._finish(asyncio.CancelledError,asyncio.CancelledError())

    def __enter__(self):return self

    def __exit__(self,error_type,error,traceback):
        try:self.close()
        except BaseException:
            if error_type is None:raise
        return False


class _TrackedAsyncChatStream:
    """Async single-consumer stream with explicit connection cleanup."""
    def __init__(self,stream,span,usage):
        self._stream=stream;self._iterator=stream.__aiter__();self._span=span;self._usage=usage
        self._closed=False;self._reading=False

    def __aiter__(self):return self

    async def __anext__(self):
        if self._closed:raise StopAsyncIteration
        if self._reading:raise ValueError('Stream supports one active reader')
        self._reading=True
        token=self._span.trace._parent.set(self._span.index)
        try:
            chunk=await self._iterator.__anext__()
            if not self._span.trace._closed and self._span.trace.spans[self._span.index]['status']=='running':
                self._usage(self._span,chunk)
            return chunk
        except StopAsyncIteration:
            self._closed=True
            try:await self._stream.close()
            except BaseException as error:
                self._span._finish(type(error),error)
                raise
            self._span._finish()
            raise
        except BaseException as error:
            self._closed=True
            try:await self._stream.close()
            except BaseException:pass
            self._span._finish(type(error),error)
            raise
        finally:
            self._span.trace._parent.reset(token)
            self._reading=False

    async def close(self):
        if self._closed:return
        self._closed=True
        try:await self._stream.close()
        finally:self._span._finish(asyncio.CancelledError,asyncio.CancelledError())

    async def aclose(self):await self.close()

    async def __aenter__(self):return self

    async def __aexit__(self,error_type,error,traceback):
        try:await self.close()
        except BaseException:
            if error_type is None:raise
        return False


class _ExternalTrace:
    def __init__(self, client, project_id, correlation_id, name, idempotency_key=None):
        self.client, self.project_id, self.correlation_id = client, project_id, correlation_id
        self.spans, self.receipt, self.export_error = [], None, None
        self._idempotency_key = idempotency_key
        self._snapshot = None
        self._export_lock = threading.Lock()
        self._lock, self._closed = threading.Lock(), False
        self._unfinished = False
        self.evaluation_errors = []
        self._parent = contextvars.ContextVar('allpaka_span_parent', default=None)
        self.root = self.span(name, 'agent')

    def _elapsed(self):
        return (time.monotonic_ns()-self._start_ns)//1_000_000

    def span(self, name, kind='tool', *, provider_id=None):
        if (kind not in ('agent', 'model', 'tool') or not isinstance(name, str) or not 1 <= len(name) <= 100
                or any(not (c.isascii() and (c.isalnum() or c in '._-:/')) for c in name)):
            raise ValueError('Use a bounded technical span identifier and supported kind')
        if provider_id is not None and (not isinstance(provider_id,str) or not 1 <= len(provider_id) <= 100
                or any(not (c.isascii() and (c.isalnum() or c in '._-:/')) for c in provider_id)):
            raise ValueError('Use a bounded technical provider identifier')
        return _TraceSpan(self, name, kind, provider_id)

    def __enter__(self):
        if self.spans or self._closed:
            raise ValueError('Trace contexts cannot be reused')
        self.started_ms = time.time_ns()//1_000_000
        self._start_ns = time.monotonic_ns()
        self.root.__enter__()
        self._active_token = _ACTIVE_TRACE.set(self)
        return self

    def __exit__(self, error_type, error, traceback):
        _ACTIVE_TRACE.reset(self._active_token)
        self.root.__exit__(error_type, error, traceback)
        unfinished = self._unfinished
        with self._lock:
            end = self._elapsed()
            for span in self.spans:
                if span['status'] == 'running':
                    unfinished = True
                    span['status'] = 'interrupted'
                    span['duration_ms'] = end-(span['started_ms']-self.started_ms)
            if unfinished:
                self.spans[0]['status'] = 'interrupted'
                self.spans[0]['duration_ms'] = end
            self._closed = True
        self._snapshot = json.dumps(dict(project_id=self.project_id, correlation_id=self.correlation_id,
            started_ms=self.started_ms, spans=self.spans), allow_nan=False)
        try:
            self.export()
        except (EvaluationError, ValueError, OSError) as export_error:
            self.export_error = export_error
            if error_type is None:
                raise
        if unfinished and error_type is None:
            raise ValueError('Trace ended with active children; interrupted evidence retained')
        return False


    def export(self):
        """Explicitly send the frozen completed snapshot; never rerun traced code.

        A retry requires a key selected before execution, because a lost response
        may already have committed the trace. Successful sends reuse their receipt.
        """
        with self._export_lock:
            if self._snapshot is None:
                raise ValueError('Export requires a completed trace')
            if self.receipt is not None:
                return self.receipt
            if self.export_error is not None and self._idempotency_key is None:
                raise ValueError('Retry requires an idempotency key selected before execution')
            snapshot = json.loads(self._snapshot)
            kwargs = {} if self._idempotency_key is None else dict(idempotency_key=self._idempotency_key)
            try:
                receipt = self.client.ingest_trace(snapshot['project_id'], snapshot['correlation_id'],
                    snapshot['started_ms'], snapshot['spans'], **kwargs)
            except (EvaluationError, ValueError, OSError) as error:
                self.export_error = error
                raise
            self.receipt, self.export_error = receipt, None
            return receipt


class Studio:
    def background_tasks(self, session_id):
        """List session-owned task metadata without launching commands."""
        return self.request('sessions/'+self._id(session_id)+'/background', {'action':'list'})

    def export_background(self, session_id, *, include_outputs=False):
        """Export owned command lifecycle receipts; captured output is opt-in."""
        if type(include_outputs) is not bool:raise ValueError('Choose explicitly whether to export command output')
        return self.request('sessions/'+self._id(session_id)+'/background',
                            {'action':'export','include_outputs':include_outputs})

    def start_background(self, session_id, command, *, timeout=3600, follow_up=None, name=None):
        """Explicitly execute a command in the session project (Auto/Goal required)."""
        if not isinstance(command,str) or not command.strip() or len(command.encode('utf-8'))>16000:
            raise ValueError('Use a nonempty command of at most 16000 UTF-8 bytes')
        if type(timeout) is not int or not 1<=timeout<=86400:
            raise ValueError('Command timeout must be 1-86400 seconds')
        body={'action':'start','command':command,'timeout':timeout}
        if name is not None:
            if not isinstance(name,str) or not name.strip() or len(name.encode('utf-8'))>200:raise ValueError('Use a task name of 1-200 UTF-8 bytes')
            body['name']=name
        if follow_up is not None:
            if not isinstance(follow_up,str) or not follow_up.strip() or len(follow_up.encode('utf-8'))>16000:raise ValueError('Use a bounded explicit follow-up')
            body['follow_up']=follow_up
        return self.request('sessions/'+self._id(session_id)+'/background',body)

    def background_output(self, session_id, task_id):
        """Fetch intentionally captured stdout/stderr and terminal metadata."""
        return self.request('sessions/'+self._id(session_id)+'/background',
                            {'action':'output','task_id':self._id(task_id)})

    def wait_background(self, session_id, task_ids, *, wait_seconds=60):
        """Wait until any selected task is terminal; never restart a task."""
        if not isinstance(task_ids,(list,tuple)) or not 1<=len(task_ids)<=32:
            raise ValueError('Select 1-32 background tasks')
        ids=[self._id(value) for value in task_ids]
        if len(set(ids))!=len(ids):raise ValueError('Select unique background task IDs')
        if type(wait_seconds) is not int or not 0<=wait_seconds<=60:
            raise ValueError('Wait must be 0-60 seconds')
        return self.request('sessions/'+self._id(session_id)+'/background',
                            {'action':'wait','task_ids':ids,'wait_seconds':wait_seconds},
                            timeout=max(10,wait_seconds+5))

    def cancel_background(self, session_id, task_id):
        """Request cancellation of an owned task; inspect the returned status."""
        return self.request('sessions/'+self._id(session_id)+'/background',
                            {'action':'cancel','task_id':self._id(task_id)})

    def cleanup_background(self, session_id, task_id=None):
        """Explicitly remove one owned terminal receipt, or all terminal receipts."""
        body={'action':'cleanup'}
        if task_id is not None:body['task_id']=self._id(task_id)
        return self.request('sessions/'+self._id(session_id)+'/background',body)

    def guardrail_policies(self, *, offset=0, limit=20):
        if type(offset) is not int or not 0<=offset<=10000 or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Use bounded policy pagination')
        result=self.request('guardrail-policies?'+urllib.parse.urlencode(dict(offset=offset,limit=limit)))
        rows=result.get('policies')
        if (result.get('provider_calls') != 0 or result.get('offset') != offset
                or result.get('limit') != limit or result.get('order') != 'policy_sha256_ascending'
                or type(result.get('total')) is not int or result['total']<0
                or not isinstance(rows,list) or len(rows)!=min(limit,max(0,result['total']-offset))):
            raise ValueError('Incompatible policy catalog')
        hashes=[]
        for row in rows:
            if (not isinstance(row,dict) or set(row)!={'policy_sha256','schema_version','rule_count'}
                    or row['schema_version']!=1 or type(row['schema_version']) is not int
                    or type(row['rule_count']) is not int or not 1<=row['rule_count']<=32
                    or not isinstance(row['policy_sha256'],str) or len(row['policy_sha256'])!=64
                    or any(c not in '0123456789abcdef' for c in row['policy_sha256'])):
                raise ValueError('Incompatible policy catalog row')
            hashes.append(row['policy_sha256'])
        if hashes!=sorted(set(hashes)):
            raise ValueError('Incompatible policy catalog ordering')
        return result

    def save_online_evaluation_rule(self,rule):
        """Save an immutable definition; this does not activate an evaluator worker."""
        if not isinstance(rule,dict) or set(rule)!={'id','project_id','evaluator_id','evaluator_version','sample_rate','enabled'}:raise ValueError('Invalid online evaluation rule')
        for key in ['id','project_id','evaluator_id']:
            value=rule[key]
            if not isinstance(value,str) or not 1<=len(value)<=100 or any(not (c.isascii() and (c.isalnum() or c in '._-')) for c in value):raise ValueError('Invalid rule identity')
        if type(rule['evaluator_version']) is not int or not 1<=rule['evaluator_version']<=2**64-1 or type(rule['enabled']) is not bool or type(rule['sample_rate']) not in (int,float) or not 0<=rule['sample_rate']<=1:raise ValueError('Invalid rule sampling or evaluator version')
        return self.request('observability/online-rules',dict(rule,sample_rate=float(rule['sample_rate'])))

    def save_online_quality_source(self,project_id,trace_id,trace_sha256,input_text,output,reference=None):
        """Persist explicit text; selected pinned model rules may automatically queue assessments."""
        project_id=self._id(project_id);trace_id=self._id(trace_id)
        if not isinstance(trace_sha256,str) or len(trace_sha256)!=64 or any(c not in '0123456789abcdef' for c in trace_sha256):raise ValueError('Invalid trace fingerprint')
        if not isinstance(input_text,str) or len(input_text.encode('utf-8'))>16000 or not isinstance(output,str) or not output.strip() or len(output.encode('utf-8'))>64000 or (reference is not None and (not isinstance(reference,str) or len(reference.encode('utf-8'))>64000)):raise ValueError('Invalid quality source text bounds')
        return self.request('observability/online-quality-sources',dict(project_id=project_id,trace_id=trace_id,trace_sha256=trace_sha256,input=input_text,output=output,reference=reference))

    def save_online_model_evaluator(self,settings,rubric):
        """Save immutable model/rubric configuration; does not activate a rule."""
        if not isinstance(rubric,str) or not rubric.strip() or len(rubric.encode('utf-8'))>16000:raise ValueError('Invalid model evaluator rubric')
        return self.request('observability/online-model-evaluators',dict(settings=settings,rubric=rubric))

    def online_model_evaluator(self,project_id,evaluator_sha256):
        project_id=self._id(project_id)
        if not isinstance(evaluator_sha256,str) or len(evaluator_sha256)!=64 or any(c not in '0123456789abcdef' for c in evaluator_sha256):raise ValueError('Invalid evaluator hash')
        return self.request('observability/online-model-evaluators/'+evaluator_sha256+'?'+urllib.parse.urlencode(dict(project_id=project_id)))

    def submit_online_quality_job(self,source_sha256,settings,rubric):
        """Queue one durable model assessment; identical requests reuse their job."""
        if not isinstance(source_sha256,str) or len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):raise ValueError('Invalid quality source fingerprint')
        if not isinstance(rubric,str) or not rubric.strip() or len(rubric.encode('utf-8'))>16000:raise ValueError('Invalid quality rubric')
        return self.request('observability/online-quality-jobs',dict(source_sha256=source_sha256,settings=settings,rubric=rubric))

    def online_quality_jobs(self,project_id,*,offset=0,limit=20):
        """Browse background model job metadata without executing assessments."""
        project_id=self._id(project_id)
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:raise ValueError('Invalid quality job page')
        return self.request('observability/online-quality-jobs?'+urllib.parse.urlencode(dict(project_id=project_id,offset=offset,limit=limit)))

    def online_quality_job(self,job_id):
        if not isinstance(job_id,str) or len(job_id)!=64 or any(c not in '0123456789abcdef' for c in job_id):raise ValueError('Invalid quality job ID')
        return self.request('observability/online-quality-jobs/'+job_id)

    def judge_online_quality_source(self,source_sha256,settings,rubric):
        """Evaluate exactly the retained text source using an explicit model/rubric."""
        if not isinstance(source_sha256,str) or len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):raise ValueError('Invalid quality source fingerprint')
        return self.request('observability/online-quality-sources/'+source_sha256+'/judge',dict(settings=settings,rubric=rubric),timeout=65)

    def online_quality_source(self,project_id,source_sha256):
        """Read explicitly retained quality text and its immutable provenance."""
        project_id=self._id(project_id)
        if not isinstance(source_sha256,str) or len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):raise ValueError('Invalid quality source fingerprint')
        return self.request('observability/online-quality-sources/'+source_sha256+'?'+urllib.parse.urlencode(dict(project_id=project_id)))

    def online_evaluation_jobs(self,project_id,*,offset=0,limit=20):
        """Read retained job states and verified result evidence without execution."""
        project_id=self._id(project_id)
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:raise ValueError('Invalid online job page')
        return self.request('observability/online-jobs?'+urllib.parse.urlencode(dict(project_id=project_id,offset=offset,limit=limit)))

    def drain_online_evaluation_jobs(self,*,limit=20):
        """Explicitly process a bounded batch of retained native evaluation jobs."""
        if type(limit) is not int or not 1<=limit<=100:raise ValueError('Invalid online job batch limit')
        return self.request('observability/online-jobs/drain',dict(limit=limit))

    def assess_online_trace(self,project_id,trace_id):
        """Execute supported native metadata evaluators for pinned selected rules."""
        return self.request('observability/online-assessments',dict(project_id=self._id(project_id),trace_id=self._id(trace_id)))

    def online_evaluation_selection_archive(self,project_id,trace_id):
        """Read retained current/legacy selection history without migration or execution."""
        return self.request('observability/online-selections?'+urllib.parse.urlencode(dict(project_id=self._id(project_id),trace_id=self._id(trace_id))))

    def select_online_evaluations(self,project_id,trace_id):
        """Persist rule selection for a completed native trace; does not execute evaluators."""
        return self.request('observability/online-selections',dict(project_id=self._id(project_id),trace_id=self._id(trace_id)))

    def bind_online_evaluation_rule(self,fingerprint,*,base_version,active):
        """Persist an optimistic rule binding used by automatic trace admission."""
        if not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint) or type(base_version) is not int or not 0<=base_version<1000 or type(active) is not bool:raise ValueError('Invalid online rule binding')
        return self.request('observability/online-rule-bindings',dict(rule_sha256=fingerprint,base_version=base_version,active=active))

    def online_evaluation_binding(self,project_id,rule_id):
        return self.request('observability/online-rule-bindings?'+urllib.parse.urlencode(dict(project_id=self._id(project_id),rule_id=self._id(rule_id))))

    def online_evaluation_rules(self,project_id,*,offset=0,limit=20):
        """Browse immutable definitions; catalog entries are not active worker bindings."""
        project_id=self._id(project_id)
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:raise ValueError('Invalid online rule page')
        return self.request('observability/online-rules?'+urllib.parse.urlencode(dict(project_id=project_id,offset=offset,limit=limit)))

    def online_evaluation_rule(self,fingerprint):
        """Read a verified retained rule snapshot without invoking an evaluator."""
        if not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):raise ValueError('Invalid rule hash')
        return self.request('observability/online-rules/'+fingerprint)

    def check_guardrail_policy(self, fingerprint, text, *, stage, action):
        if (not isinstance(fingerprint,str) or len(fingerprint)!=64
                or any(c not in '0123456789abcdef' for c in fingerprint)):
            raise ValueError('Use a SHA-256 policy fingerprint')
        if not isinstance(text,str) or len(text.encode('utf-8'))>64000:
            raise ValueError('Use text up to 64000 UTF-8 bytes')
        if stage not in ('input','output') or action not in ('observe','block'):
            raise ValueError('Choose explicit guardrail stage and action')
        result=self.request('guardrail-policies/'+fingerprint+'/check',dict(text=text,stage=stage,action=action))
        if (result.get('kind')!='local_guardrail' or result.get('schema_version')!=1
                or result.get('policy_sha256')!=fingerprint or result.get('stage')!=stage
                or result.get('action')!=action or result.get('provider_calls')!=0
                or result.get('content_captured') is not False
                or type(result.get('passed')) is not bool
                or result.get('blocked') is not (not result['passed'] and action=='block')):
            raise ValueError('Incompatible guardrail check receipt')
        rules=result.get('rules')
        if (not isinstance(rules,list) or not 1<=len(rules)<=32
                or any(not isinstance(rule,dict) or set(rule)!={'rule_id','kind','passed'}
                       or type(rule['passed']) is not bool or not isinstance(rule['rule_id'],str)
                       or rule['kind'] not in ('min_bytes','max_bytes','json_valid','forbidden_substrings','required_substrings') for rule in rules)
                or len({rule['rule_id'] for rule in rules})!=len(rules)
                or all(rule['passed'] for rule in rules)!=result['passed']):
            raise ValueError('Incompatible guardrail rule outcomes')
        return result

    def create_guardrail_policy(self, rules):
        from allpaka_guardrails import policy_manifest, rules_from_manifest
        expected=policy_manifest(rules)
        result=self.request('guardrail-policies/create',dict(rules=expected['rules']))
        if result.get('provider_calls') != 0 or result.get('policy') != expected:
            raise ValueError('Incompatible created policy receipt')
        rules_from_manifest(result['policy'])
        return result

    def save_guardrail_policy(self, rules):
        from allpaka_guardrails import policy_manifest, rules_from_manifest
        manifest=policy_manifest(rules)
        result=self.request('guardrail-policies',manifest)
        if result.get('provider_calls') != 0 or result.get('policy') != manifest:
            raise ValueError('Incompatible saved policy receipt')
        rules_from_manifest(result['policy'])
        return result

    def guardrail_policy(self, fingerprint):
        from allpaka_guardrails import rules_from_manifest
        if (not isinstance(fingerprint,str) or len(fingerprint)!=64
                or any(c not in '0123456789abcdef' for c in fingerprint)):
            raise ValueError('Use a SHA-256 policy fingerprint')
        result=self.request('guardrail-policies/'+fingerprint)
        rules_from_manifest(result.get('policy'))
        if result.get('provider_calls') != 0 or result['policy']['policy_sha256'] != fingerprint:
            raise ValueError('Incompatible policy receipt')
        return result

    def guard_task_with_policies(self, task, *, input_policy_sha256,
                                 output_policy_sha256, action, trace=None,
                                 return_receipts=True):
        """Resolve pinned stored rules before executing a sync/async task."""
        from allpaka_guardrails import guard_task, rules_from_manifest
        if action not in ('block','observe'):
            raise ValueError('Choose block or observe explicitly')
        if not callable(task):
            raise ValueError('Use a callable task')
        input_manifest=self.guardrail_policy(input_policy_sha256)['policy']
        output_manifest=self.guardrail_policy(output_policy_sha256)['policy']
        return guard_task(task,input_rules=rules_from_manifest(input_manifest),
                          output_rules=rules_from_manifest(output_manifest),
                          action=action,trace=trace,return_receipts=return_receipts)

    def create_review_queue(self, project_id, name, targets, *, instructions=''):
        """Persist explicit trace/span review sources; never call a model."""
        project_id=self._id(project_id)
        if not isinstance(name,str) or not name.strip() or len(name.encode('utf-8'))>200 or not isinstance(instructions,str) or len(instructions.encode('utf-8'))>16000:
            raise ValueError('Use bounded review queue name and instructions')
        if not isinstance(targets,list) or not 1<=len(targets)<=200:raise ValueError('Select 1-200 review targets')
        frozen=[];seen=set()
        for target in targets:
            if not isinstance(target,dict) or set(target)-{'trace_id','span_id'}:raise ValueError('Use trace/span review references')
            trace=self._id(target.get('trace_id'));span=target.get('span_id')
            if span is not None and (type(span) is not int or not 0<=span<2**64):raise ValueError('Invalid review span')
            key=(trace,span)
            if key in seen:raise ValueError('Duplicate review source')
            seen.add(key);frozen.append(dict(trace_id=trace,span_id=span))
        return self.request('observability/review-queues',dict(project_id=project_id,name=name,instructions=instructions,targets=frozen))

    def assign_review_queue(self, queue_id, target_index, reviewer, *, base_version):
        """Set/clear a self-reported reviewer; conflicts are never retried."""
        if type(target_index) is not int or not 0<=target_index<200 or type(base_version) is not int or not 1<=base_version<2**64:
            raise ValueError('Use a review target index and current queue version')
        if reviewer is not None and (not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer.encode('utf-8'))>200):raise ValueError('Use a bounded reviewer name or None to clear')
        return self.request('observability/review-queues/'+self._id(queue_id)+'/assignments',dict(base_version=base_version,target_index=target_index,reviewer=reviewer))

    def _review_completion(self, queue_id, target_index, base_version, action, **evidence):
        if type(target_index) is not int or not 0<=target_index<200 or type(base_version) is not int or not 1<=base_version<2**64:
            raise ValueError('Use a review target index and current queue version')
        return self.request('observability/review-queues/'+self._id(queue_id)+'/completion',dict(base_version=base_version,target_index=target_index,action=action,**evidence))

    def complete_review_queue(self, queue_id, target_index, feedback_version, annotation_id, *, base_version):
        """Complete against a pinned saved review; this is not a quality promotion."""
        if type(feedback_version) is not int or not 1<=feedback_version<=2000:raise ValueError('Select a saved feedback version 1-2000')
        return self._review_completion(queue_id,target_index,base_version,'complete',feedback_version=feedback_version,annotation_id=self._id(annotation_id))

    def reopen_review_queue(self, queue_id, target_index, *, base_version):
        """Explicitly clear completion while retaining the assigned reviewer."""
        return self._review_completion(queue_id,target_index,base_version,'reopen')

    def review_queue(self, queue_id):
        return self.request('observability/review-queues/'+self._id(queue_id))

    def review_queue_history(self,queue_id,*,offset=0,limit=20):
        """Read append-only local change entries; old queues may have partial history."""
        if type(offset) is not int or not 0<=offset<=2000 or type(limit) is not int or not 1<=limit<=100:raise ValueError('Use history offset 0-2000 and limit 1-100')
        return self.request('observability/review-queues/'+self._id(queue_id)+'/history?'+urllib.parse.urlencode(dict(offset=offset,limit=limit)))

    def set_review_queue_archived(self,queue_id,archived,*,base_version):
        """Archive or restore without deleting sources, assignments or evidence."""
        if type(archived) is not bool or type(base_version) is not int or not 1<=base_version<=2**64-1:raise ValueError('Use boolean archive state and positive base version')
        return self.request('observability/review-queues/'+self._id(queue_id)+'/lifecycle',dict(base_version=base_version,archived=archived))

    def export_review_queue_csv(self, queue_id):
        """Export one saved queue snapshot, including immutable feedback references.

        Does not fetch trace content or feedback text and makes no model calls.
        Reviewer names are self-reported. CSV version identifies this snapshot.
        """
        queue=self.review_queue(queue_id)
        def invalid():raise EvaluationError('invalid_review_queue_export')
        if not isinstance(queue,dict) or queue.get('id')!=queue_id or type(queue.get('version')) is not int or not 1<=queue['version']<=2**64-1:invalid()
        if not isinstance(queue.get('project_id'),str) or not queue['project_id'] or not isinstance(queue.get('name'),str):invalid()
        targets=queue.get('targets');assignments=queue.get('assignments');completions=queue.get('completions')
        if not isinstance(targets,list) or not 1<=len(targets)<=200 or not isinstance(assignments,dict) or not isinstance(completions,dict):invalid()
        keys={str(i) for i in range(len(targets))}
        if set(assignments)-keys or set(completions)-keys:invalid()
        def cell(value):
            if value is None:return ''
            if type(value) is int:return value
            if not isinstance(value,str):invalid()
            return "'"+value if value.lstrip().startswith(('=','+','-','@')) or value.startswith(('\t','\r','\n')) else value
        stream=io.StringIO(newline='');writer=csv.writer(stream)
        writer.writerow(['queue_id','queue_version','project_id','queue_name','archived','target_index','trace_id','span_id','status','reviewer','feedback_version','annotation_id'])
        seen=set()
        for index,target in enumerate(targets):
            if not isinstance(target,dict) or not isinstance(target.get('trace_id'),str) or not target['trace_id']:invalid()
            span=target.get('span_id')
            if span is not None and (type(span) is not int or not 0<=span<=2**64-1):invalid()
            identity=(target['trace_id'],span)
            if identity in seen:invalid()
            seen.add(identity);reviewer=assignments.get(str(index));pin=completions.get(str(index))
            if reviewer is not None and (not isinstance(reviewer,str) or not reviewer.strip()):invalid()
            if pin is not None and (not isinstance(pin,dict) or reviewer is None or pin.get('reviewer')!=reviewer or type(pin.get('feedback_version')) is not int or not 1<=pin['feedback_version']<=2000 or not isinstance(pin.get('annotation_id'),str) or not pin['annotation_id']):invalid()
            writer.writerow([cell(v) for v in [queue_id,queue['version'],queue['project_id'],queue['name'],'true' if queue.get('archived',False) else 'false',index,target['trace_id'],span,'completed' if pin else 'assigned' if reviewer else 'unassigned',reviewer,pin['feedback_version'] if pin else None,pin['annotation_id'] if pin else None]])
        return stream.getvalue()

    def review_queues(self, project_id, *, offset=0, limit=20, status=None, reviewer=None, name=None, archived=None):
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:raise ValueError('Use review queue offset 0-1000 and limit 1-100')
        if archived is not None and type(archived) is not bool:raise ValueError('Use boolean archive filter')
        if name is not None and (not isinstance(name,str) or not name.strip() or len(name.encode('utf-8'))>200):raise ValueError('Invalid queue name search')
        if status is not None and status not in ('pending','completed','unassigned'):raise ValueError('Invalid review queue status')
        if reviewer is not None and (not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer.encode('utf-8'))>200):raise ValueError('Invalid reviewer name')
        query=dict(project_id=self._id(project_id),offset=offset,limit=limit)
        if status is not None:query['status']=status
        if reviewer is not None:query['reviewer']=reviewer
        if name is not None:query['name']=name
        if archived is not None:query['archived']='true' if archived else 'false'
        return self.request('observability/review-queues?'+urllib.parse.urlencode(query))

    def feedback(self, trace_id, *, version=None):
        """Read current or immutable historical human annotations."""
        if version is not None and (type(version) is not int or not 0<=version<=2000):
            raise ValueError('Use feedback version 0-2000')
        path='observability/traces/'+self._id(trace_id)+'/feedback'
        return self.request(path+('' if version is None else '?'+urllib.parse.urlencode(dict(version=version))))

    def save_feedback(self, trace_id, annotation, *, base_version):
        """Append a human-authored review revision; reviewer identity is self-reported.

        Reuse the returned annotation ID with deleted=True/False for removal or
        restoration. Conflicts are returned by the server, never auto-retried.
        """
        trace_id=self._id(trace_id)
        if type(base_version) is not int or not 0<=base_version<2000:
            raise ValueError('Use the current feedback base version below 2000')
        if not isinstance(annotation,dict) or set(annotation)-{'id','span_id','author','metric','value','category','comment','correction','deleted'}:
            raise ValueError('Use a feedback annotation with known fields')
        try: frozen=json.loads(json.dumps(annotation,allow_nan=False))
        except (TypeError,ValueError) as error:raise ValueError('Use JSON annotation values') from error
        author=frozen.get('author')
        if not isinstance(author,str) or not author.strip() or len(author.encode('utf-8'))>200:
            raise ValueError('Provide a reviewer name within 200 UTF-8 bytes')
        annotation_id=frozen.get('id')
        if annotation_id is not None and (not isinstance(annotation_id,str) or not 1<=len(annotation_id)<=100 or any(not (c.isascii() and (c.isalnum() or c=='-')) for c in annotation_id)):
            raise ValueError('Invalid annotation ID')
        span=frozen.get('span_id')
        if span is not None and (type(span) is not int or not 0<=span<2**64):raise ValueError('Invalid review span')
        if 'deleted' in frozen and type(frozen['deleted']) is not bool:raise ValueError('Use a boolean removal flag')
        for field,limit in [('metric',100),('category',200),('comment',16000),('correction',65536)]:
            value=frozen.get(field)
            if value is not None and (not isinstance(value,str) or len(value.encode('utf-8'))>limit or field in ('metric','category') and not value.strip()):
                raise ValueError('Invalid feedback '+field)
        value=frozen.get('value');category=frozen.get('category');metric=frozen.get('metric')
        if value is not None and (type(value) not in (int,float) or abs(value)>1000000 or not math.isfinite(value)):raise ValueError('Invalid numeric review score')
        if value is not None and category is not None or (metric is not None)!= (value is not None or category is not None):raise ValueError('Provide one named numeric or categorical score')
        if metric is None and not (frozen.get('comment') or '').strip() and not frozen.get('correction'):raise ValueError('Provide a score, comment or corrected answer')
        return self.request('observability/traces/'+trace_id+'/feedback',dict(base_version=base_version,annotation=frozen))

    def feedback_versions(self, trace_id, *, offset=0, limit=20):
        """Metadata-only pages of immutable human review revisions."""
        if type(offset) is not int or not 0<=offset<=2000 or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Use feedback history offset 0-2000 and limit 1-100')
        return self.request('observability/traces/'+self._id(trace_id)+'/feedback/versions?'+
                            urllib.parse.urlencode(dict(offset=offset,limit=limit)))

    def export_trace(self, trace_id, *, include_feedback=False):
        if type(include_feedback) is not bool:
            raise ValueError('Choose explicitly whether to export human feedback')
        return self.request('observability/traces/'+self._id(trace_id)+'/export?'+
                            urllib.parse.urlencode(dict(include_feedback=str(include_feedback).lower())))

    def export_traces(self, project_id, trace_ids, *, include_feedback=False):
        """Export 1-100 selected terminal traces; no partial batch or model calls."""
        if type(include_feedback) is not bool:raise ValueError('Choose explicitly whether to export feedback')
        if not isinstance(trace_ids,(list,tuple)) or not 1<=len(trace_ids)<=100:
            raise ValueError('Choose 1-100 unique trace IDs')
        ids=[self._id(trace_id) for trace_id in trace_ids]
        if len(set(ids))!=len(ids):raise ValueError('Choose unique trace IDs')
        return self.request('observability/trace-exports',dict(project_id=self._id(project_id),
            trace_ids=ids,include_feedback=include_feedback),timeout=30)

    def remove_trace(self, trace_id):
        return self.request('observability/traces/'+self._id(trace_id)+'/remove',{})

    def restore_trace(self, trace_id):
        return self.request('observability/traces/'+self._id(trace_id)+'/restore',{})

    def track_openai_chat(self, create, *, provider_id, model_id):
        """Wrap a non-streaming Chat Completions create callable, sync or async.

        Explicit technical model/provider IDs label spans. Only reported usage
        is inspected; prompts, choices, response IDs and errors are never saved.
        Returns the original response and adds no retries or SDK dependency.
        """
        return self._track_provider_create(create,provider_id,model_id,
            (('prompt_tokens','input_tokens'),('completion_tokens','output_tokens')),'prompt_tokens_details')

    def track_openai_chat_stream(self, create, *, provider_id, model_id):
        """Wrap sync/async streaming create calls; consume/close inside a trace.

        Yields original chunks without buffering. Exhaustion completes the span;
        early close interrupts it. Usage is recorded only when supplied in chunks.
        """
        return self._track_provider_create(create,provider_id,model_id,
            (('prompt_tokens','input_tokens'),('completion_tokens','output_tokens')),
            'prompt_tokens_details',streaming=True)

    def track_openai_responses(self, create, *, provider_id, model_id):
        """Wrap sync/async Responses create calls with metadata-only model spans.

        Requires non-streaming, foreground responses and an explicitly pinned
        model. Only reported input/output/cache token counters are collected.
        Returned responses and original exceptions are preserved.
        """
        return self._track_provider_create(create,provider_id,model_id,
            (('input_tokens','input_tokens'),('output_tokens','output_tokens')),'input_tokens_details',responses=True)

    def track_openai_responses_stream(self, create, *, provider_id, model_id):
        """Trace sync/async foreground Responses events without retaining text.

        A valid completed terminal event is required for successful completion;
        failed/incomplete events mark failure and missing terminal marks interruption.
        """
        return self._track_provider_create(create,provider_id,model_id,
            (('input_tokens','input_tokens'),('output_tokens','output_tokens')),
            'input_tokens_details',responses=True,streaming=True)

    def track_anthropic_messages(self, create, *, provider_id, model_id):
        """Trace sync/async non-streaming Messages create calls without content.

        Total input includes reported uncached, cache creation and cache read
        tokens. Missing components leave total input unknown; costs aren't guessed.
        """
        return self._track_provider_create(create,provider_id,model_id,
            (('output_tokens','output_tokens'),),None,anthropic=True)

    def track_anthropic_messages_stream(self, create, *, provider_id, model_id):
        """Trace sync/async Messages create(..., stream=True) raw event streams.

        Requires message_start/message_stop for completion. Usage updates are
        cumulative, not added across deltas; contents and errors aren't retained.
        """
        return self._track_provider_create(create,provider_id,model_id,
            (('output_tokens','output_tokens'),),None,anthropic=True,streaming=True)

    def _track_provider_create(self, create, provider_id, model_id, token_fields, detail_field, responses=False, streaming=False, anthropic=False):
        if not callable(create) or inspect.isgeneratorfunction(create) or inspect.isasyncgenfunction(create):
            raise ValueError('Choose a sync or async create callable')
        _ExternalTrace(self,'validation','validation','validation').span(model_id,'model',provider_id=provider_id)
        def active(kwargs):
            if kwargs.get('model')!=model_id:
                raise ValueError('Completion model must match the pinned adapter model')
            if streaming:
                if kwargs.get('stream') is not True:raise ValueError('Streaming adapter requires stream=True')
            elif kwargs.get('stream',False):
                raise ValueError('This adapter requires non-streaming responses')
            if responses and kwargs.get('background',False):
                raise ValueError('This adapter requires foreground responses')
            trace=_ACTIVE_TRACE.get()
            return trace if trace is not None and trace.client is self and not trace._closed else None
        def usage(span,response):
            # Missing/invalid counters remain unknown; never estimate cost.
            def field(value,name):
                try:return value.get(name) if isinstance(value,dict) else getattr(value,name,None)
                except Exception:return None
            if streaming and not getattr(span,'_first_text_seen',False):
                event=field(response,'type');text=None
                if responses and event=='response.output_text.delta':text=field(response,'delta')
                elif anthropic and event=='content_block_delta':
                    delta=field(response,'delta')
                    if field(delta,'type')=='text_delta':text=field(delta,'text')
                elif not responses and not anthropic:
                    choices=field(response,'choices')
                    if isinstance(choices,(list,tuple)):
                        for choice in choices:
                            candidate=field(field(choice,'delta'),'content')
                            if isinstance(candidate,str) and candidate:text=candidate;break
                if isinstance(text,str) and text:
                    span.set_usage(first_text_ms=span.trace._elapsed()-span.start)
                    span._first_text_seen=True
            if responses and streaming:
                event_type=field(response,'type')
                expected={'response.completed':'completed','response.failed':'failed','response.incomplete':'incomplete'}
                if not isinstance(event_type,str) or event_type not in expected:return
                response=field(response,'response')
                if field(response,'status')!=expected[event_type]:return
                span._terminal_seen=True
                span._provider_failure=getattr(span,'_provider_failure',False) or event_type!='response.completed'
            if anthropic and streaming:
                event_type=field(response,'type')
                if event_type=='error':
                    span._terminal_seen=True;span._provider_failure=True
                    return
                if event_type=='message_stop':
                    if getattr(span,'_anthropic_started',False):span._terminal_seen=True
                    return
                if event_type=='message_start':
                    message=field(response,'message')
                    if field(message,'type')!='message':return
                    if getattr(span,'_anthropic_started',False):span._provider_failure=True
                    span._anthropic_started=True;span._anthropic_inputs={}
                    response=message
                elif event_type=='message_delta':
                    if not getattr(span,'_anthropic_started',False):return
                else:return
            reported=field(response,'usage')
            counters={}
            for source,target in token_fields:
                value=field(reported,source)
                if type(value) is int and 0<=value<=2**64-1:counters[target]=value
            cached=None if anthropic else field(field(reported,detail_field),'cached_tokens')
            if type(cached) is int and 0<=cached<=counters.get('input_tokens',-1):
                counters['cache_read_input_tokens']=cached
            if anthropic:
                keys=('input_tokens','cache_creation_input_tokens','cache_read_input_tokens')
                if streaming:
                    for key in keys:
                        value=field(reported,key)
                        if type(value) is int and 0<=value<=2**64-1:span._anthropic_inputs[key]=value
                    inputs=[span._anthropic_inputs.get(key) for key in keys]
                else:inputs=[field(reported,key) for key in keys]
                if all(type(value) is int and 0<=value<=2**64-1 for value in inputs) and sum(inputs)<=2**64-1:
                    counters['input_tokens']=sum(inputs)
                    counters['cache_read_input_tokens']=inputs[2]
                    counters['cache_creation_input_tokens']=inputs[1]
            if counters:span.set_usage(**counters)
        if inspect.iscoroutinefunction(create) or inspect.iscoroutinefunction(getattr(create,'__call__',None)):
            @functools.wraps(create)
            async def asynchronous(*args,**kwargs):
                trace=active(kwargs)
                if trace is None:return await create(*args,**kwargs)
                if streaming:
                    span=trace.span(model_id,'model',provider_id=provider_id)
                    span._requires_terminal=responses or anthropic
                    span.__enter__()
                    response=None
                    try:
                        response=await create(*args,**kwargs)
                        return _TrackedAsyncChatStream(response,span,usage)
                    except BaseException as error:
                        if response is not None:
                            try:await response.close()
                            except BaseException:pass
                        span._finish(type(error),error)
                        raise
                    finally:trace._parent.reset(span.token)
                with trace.span(model_id,'model',provider_id=provider_id) as span:
                    response=await create(*args,**kwargs)
                    usage(span,response)
                    return response
            return asynchronous
        @functools.wraps(create)
        def synchronous(*args,**kwargs):
            trace=active(kwargs)
            if trace is None:return create(*args,**kwargs)
            if streaming:
                span=trace.span(model_id,'model',provider_id=provider_id)
                span._requires_terminal=responses or anthropic
                span.__enter__()
                response=None
                try:
                    response=create(*args,**kwargs)
                    return _TrackedChatStream(response,span,usage)
                except BaseException as error:
                    if response is not None:
                        try:response.close()
                        except BaseException:pass
                    span._finish(type(error),error)
                    raise
                finally:trace._parent.reset(span.token)
            with trace.span(model_id,'model',provider_id=provider_id) as span:
                response=create(*args,**kwargs)
                usage(span,response)
                return response
        return synchronous

    def track(self, name, kind='tool', *, provider_id=None, evaluate=None, evaluation_sample_rate=1.0, evaluator_id=None, evaluator_version=None):
        """Trace a sync/async function only inside this client's active trace.

        Default tracing never inspects arguments, results or exception text.
        Explicit evaluate(result) may return 1-20 numeric scores (0-1), stored
        in a nested evaluation span. Evaluator exceptions mark that span failed
        and are retained locally in trace.evaluation_errors; the task result
        is preserved. No evaluation occurs outside an active trace. Generator
        functions require explicit span blocks around iteration instead.
        evaluation_sample_rate (default 1) deterministically samples using a
        versioned SHA-256 key of project/correlation/name/span index. Rate 0
        skips every evaluator; skipped calls produce no quality score. Reuse
        correlation IDs and trace structure only when repeatable selection is intended.
        """
        if evaluator_id is not None or evaluator_version is not None:
            if evaluate is None or not isinstance(evaluator_id,str) or not 1<=len(evaluator_id)<=100 or any(not (c.isascii() and (c.isalnum() or c in '._-')) for c in evaluator_id) or type(evaluator_version) is not int or not 1<=evaluator_version<=2**64-1:raise ValueError('Evaluator identity requires a callback, technical ID and positive version')
        if type(evaluation_sample_rate) not in (int,float) or not 0<=evaluation_sample_rate<=1 or not math.isfinite(evaluation_sample_rate):raise ValueError('Use evaluation sample rate 0-1')
        if evaluate is None and evaluation_sample_rate!=1:raise ValueError('Sampling requires an evaluator')
        if evaluate is not None and not callable(evaluate):raise ValueError('Use a callable evaluator')
        def selected(trace,span):
            key=json.dumps(['allpaka-evaluation-sampling-v1',trace.project_id,trace.correlation_id,name,span.index],separators=(',',':'),ensure_ascii=False).encode('utf-8')
            chosen=evaluation_sample_rate==1 or evaluation_sample_rate>0 and int.from_bytes(hashlib.sha256(key).digest()[:8],'big') < math.ceil(evaluation_sample_rate*2**64)
            with trace._lock:trace.spans[span.index]['usage']['evaluation_sampling']=dict(method='sha256_v1',sample_rate=evaluation_sample_rate,selected=chosen)
            return chosen
        _ExternalTrace(self, 'validation', 'validation', name).span(name, kind, provider_id=provider_id)
        def decorate(function):
            if inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function):
                raise ValueError('Use explicit spans around generator iteration')
            if not inspect.iscoroutinefunction(function) and evaluate is not None and inspect.iscoroutinefunction(evaluate):raise ValueError('Async evaluator requires an async tracked function')
            if inspect.iscoroutinefunction(function):
                @functools.wraps(function)
                async def asynchronous(*args, **kwargs):
                    trace = _ACTIVE_TRACE.get()
                    if trace is None or trace.client is not self or trace._closed:
                        return await function(*args, **kwargs)
                    with trace.span(name, kind, provider_id=provider_id) as tracked_span:
                        result=await function(*args, **kwargs)
                        if evaluate is not None and selected(trace,tracked_span):
                            try:
                                with trace.span('evaluation','tool') as assessment:
                                    if evaluator_id is not None:assessment.set_evaluator_ref(evaluator_id,evaluator_version)
                                    scores=evaluate(result)
                                    if inspect.isawaitable(scores):scores=await scores
                                    assessment.set_evaluation_scores(scores)
                            except Exception as error:trace.evaluation_errors.append(error)
                        return result
                return asynchronous
            @functools.wraps(function)
            def synchronous(*args, **kwargs):
                trace = _ACTIVE_TRACE.get()
                if trace is None or trace.client is not self or trace._closed:
                    return function(*args, **kwargs)
                with trace.span(name, kind, provider_id=provider_id) as tracked_span:
                    result=function(*args, **kwargs)
                    if evaluate is not None and selected(trace,tracked_span):
                        try:
                            with trace.span('evaluation','tool') as assessment:
                                if evaluator_id is not None:assessment.set_evaluator_ref(evaluator_id,evaluator_version)
                                scores=evaluate(result)
                                if inspect.isawaitable(scores):
                                    if inspect.iscoroutine(scores):scores.close()
                                    raise ValueError('Sync evaluator cannot return awaitable')
                                assessment.set_evaluation_scores(scores)
                        except Exception as error:trace.evaluation_errors.append(error)
                    return result
            return synchronous
        return decorate

    def trace(self, project_id, correlation_id, name='pipeline', *, idempotency_key=None):
        """Context manager for external metadata; exceptions/arguments are never captured."""
        self._id(project_id)
        if (not isinstance(correlation_id, str) or not 1 <= len(correlation_id) <= 100
                or any(not (c.isascii() and (c.isalnum() or c in '._-:/')) for c in correlation_id)):
            raise ValueError('Use a bounded technical correlation identifier')
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 100
                or any(not (c.isascii() and (c.isalnum() or c in '._-:/')) for c in idempotency_key)):
            raise ValueError('Use a bounded technical idempotency key')
        return _ExternalTrace(self, project_id, correlation_id, name, idempotency_key)

    def ingest_trace(self, project_id, correlation_id, started_ms, spans, *, idempotency_key=None):
        """Persist a completed metadata-only external tree; no model invocation.

        Span IDs are list indices. Parents must precede children. This creates
        a new immutable trace unless an explicit idempotency key deduplicates it.
        """
        body=dict(project_id=project_id,correlation_id=correlation_id,started_ms=started_ms,spans=spans)
        if idempotency_key is not None:
            body['idempotency_key']=idempotency_key
        return self.request('observability/external-traces',body)

    def __init__(self, base_url="http://127.0.0.1:8100"):
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError("Use a Studio origin without credentials, path, query or fragment")
        # Validate the port before making a request.
        parsed.port
        self.base_url = base_url.rstrip("/")
        self.opener = urllib.request.build_opener(_NoRedirect())

    def request(self, path, body=None, timeout=10):
        request = urllib.request.Request(
            self.base_url + "/api/" + path,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "X-Allpaka-Client": "studio"},
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise EvaluationError("response_too_large")
                return json.loads(data, parse_constant=_reject_nonfinite)
        except urllib.error.HTTPError as error:
            # Provider/server text may contain private input; don't echo it into CI logs.
            code = error.code
            error.close()
            raise EvaluationError("http_" + str(code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise EvaluationError("connection_error") from None
        except (ValueError, UnicodeError):
            raise EvaluationError("invalid_response") from None

    @staticmethod
    def _id(value):
        if not isinstance(value, str) or not value or len(value) > 80 or any(
                not (c.isascii() and (c.isalnum() or c in "-_")) for c in value):
            raise ValueError("Invalid Studio artifact ID")
        return value

    def start_judges(self, plan_id):
        return self.request("evaluation/judge-runs",dict(plan_id=self._id(plan_id)))

    def judge_run(self, run_id, timeout=10):
        return self.request("evaluation/judge-runs/"+self._id(run_id),timeout=timeout)

    def judge_runs(self, project_id, offset=0, limit=20):
        return self.request("evaluation/judge-runs?"+urllib.parse.urlencode(dict(project_id=project_id,offset=offset,limit=limit)))

    def cancel_judges(self, run_id):
        return self.request("evaluation/judge-runs/"+self._id(run_id)+"/cancel",{})

    def compare_judges(self, baseline_id, candidate_id):
        return self.request('evaluation/judge-runs/compare',dict(baseline_id=self._id(baseline_id),candidate_id=self._id(candidate_id)))

    def wait_judges(self, run_id, timeout=300, poll_interval=0.25):
        """Observe an existing batch; this method never requests cancellation."""
        run_id=self._id(run_id)
        if not math.isfinite(timeout) or timeout<=0 or not math.isfinite(poll_interval) or poll_interval<=0:
            raise ValueError("Timeout and poll interval must be finite and positive")
        deadline=time.monotonic()+timeout
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise EvaluationError("judge_timeout",run_id)
            try:run=self.judge_run(run_id,timeout=min(10,remaining))
            except EvaluationError as error:
                if error.reason=='connection_error' and time.monotonic()>=deadline:
                    raise EvaluationError('judge_timeout',run_id) from None
                raise
            if not isinstance(run,dict) or run.get('id')!=run_id or run.get('status') not in ('running','completed','failed','cancelled','interrupted'):
                raise EvaluationError('invalid_judge_run',run_id)
            if run['status']!='running':return run
            time.sleep(min(poll_interval,max(0,deadline-time.monotonic())))

    def evaluate_judge_plan(self, plan_id, *, min_score=None, baseline_id=None, require_improvement=False, timeout=300, poll_interval=0.25):
        """Start one batch, wait and apply a threshold; cancel this batch on wait errors."""
        plan_id=self._id(plan_id)
        if not math.isfinite(timeout) or timeout<=0 or not math.isfinite(poll_interval) or poll_interval<=0:
            raise ValueError("Timeout and poll interval must be finite and positive")
        if min_score is not None and (type(min_score) not in (int,float) or not math.isfinite(min_score) or not 0<=min_score<=1):
            raise ValueError("Judge minimum must be finite from zero to one")
        if require_improvement and baseline_id is None:raise ValueError('Strict improvement requires a baseline')
        plan=self.judge_plan(plan_id)
        if plan.get('id')!=plan_id or plan.get('kind')!='judge_plan':raise EvaluationError('invalid_judge_plan')
        if baseline_id is not None:
            baseline_id=self._id(baseline_id);baseline=self.judge_run(baseline_id)
            if baseline.get('id')!=baseline_id or baseline.get('status')!='completed':raise EvaluationError('judge_baseline_not_completed',baseline_id)
            baseline_plan=self.judge_plan(self._id(baseline.get('plan_id')))
            if baseline.get('plan_sha256')!=baseline_plan.get('plan_sha256'):raise EvaluationError('invalid_judge_baseline',baseline_id)
            for field in ('project_id','dataset_id','dataset_version','dataset_sha256','rubric','rubric_snapshot','judge_preset','settings'):
                if plan.get(field)!=baseline_plan.get(field):raise EvaluationError('judge_baseline_mismatch',baseline_id)
        created=self.start_judges(plan_id);run_id=self._id(created.get('id'))
        try:
            if created.get('plan_id')!=plan_id:raise EvaluationError('invalid_judge_run',run_id)
            run=self.wait_judges(run_id,timeout,poll_interval)
            if run.get('plan_id')!=plan_id or run.get('plan_sha256')!=plan.get('plan_sha256'):
                raise EvaluationError('invalid_judge_run',run_id)
            score=run.get('mean_score')
            if run['status']=='completed' and (type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1):
                raise EvaluationError('invalid_metric_receipt',run_id)
            passed=run['status']=='completed' and (min_score is None or score>=min_score)
            comparison=None
            if run['status']=='completed' and baseline_id is not None:
                comparison=self.compare_judges(baseline_id,run_id)
                regressions=comparison.get('regressions');improvements=comparison.get('improvements')
                if (comparison.get('kind')!='judge_comparison' or comparison.get('baseline_id')!=baseline_id
                        or comparison.get('candidate_id')!=run_id or type(regressions) is not int
                        or type(improvements) is not int or regressions<0 or improvements<0
                        or regressions+improvements>len(run.get('items',[]))
                        or type(comparison.get('eligible')) is not bool
                        or comparison['eligible']!=(regressions==0 and improvements>0)):
                    raise EvaluationError('invalid_judge_comparison',run_id)
                passed=passed and regressions==0 and (not require_improvement or comparison['eligible'])
            return dict(passed=passed,run=run,min_score=min_score,observed=score,comparison=comparison,
                        require_improvement=require_improvement,automatic_promotion=False)
        except (EvaluationError,KeyboardInterrupt) as error:
            cancellation_failed=False
            try:self.cancel_judges(run_id)
            except (EvaluationError,ValueError):cancellation_failed=True
            if isinstance(error,KeyboardInterrupt):
                error.run_id=run_id;error.cancellation_failed=cancellation_failed;raise
            raise EvaluationError(error.reason,run_id,cancellation_failed) from None

    def judge_presets(self):
        return self.request('evaluation/judge-presets')

    def provider_doctor(self, provider_id, model=None):
        """Explicit catalog-only connection probe; never invokes inference."""
        return self.request('providers/'+self._id(provider_id)+'/doctor',dict(model=model),timeout=15)

    def traces(self, *, project_id=None, session_id=None, guardrail=None, guardrail_policy_sha256=None, status=None, since_ms=None, until_ms=None, removed=False, offset=0, limit=50):
        """Browse persisted metadata; guardrail filters select reported SDK outcomes."""
        if guardrail is not None and guardrail not in ('any','failed','blocked'):
            raise ValueError('Invalid guardrail trace filter')
        if guardrail_policy_sha256 is not None and (not isinstance(guardrail_policy_sha256,str) or len(guardrail_policy_sha256)!=64 or any(c not in '0123456789abcdef' for c in guardrail_policy_sha256)):
            raise ValueError('Invalid guardrail policy SHA-256')
        if type(removed) is not bool or type(offset) is not int or not 0<=offset<=100000 or type(limit) is not int or not 1<=limit<=200:
            raise ValueError('Invalid trace pagination or visibility')
        if status is not None and (not isinstance(status,str) or not 1<=len(status)<=40 or any(c not in 'abcdefghijklmnopqrstuvwxyz_' for c in status)):
            raise ValueError('Invalid trace status filter')
        for value in (since_ms,until_ms):
            if value is not None and (type(value) is not int or not 0<=value<=2**64-1):raise ValueError('Invalid trace time boundary')
        if since_ms is not None and until_ms is not None and since_ms>until_ms:raise ValueError('Invalid trace time range')
        query=dict(removed=str(removed).lower(),offset=offset,limit=limit)
        for name,value in (('project_id',project_id),('session_id',session_id)):
            if value is not None:query[name]=self._id(value)
        if guardrail is not None:query['guardrail']=guardrail
        if guardrail_policy_sha256 is not None:query['guardrail_policy_sha256']=guardrail_policy_sha256
        for name,value in (('status',status),('since_ms',since_ms),('until_ms',until_ms)):
            if value is not None:query[name]=value
        return self.request('observability/traces?'+urllib.parse.urlencode(query))

    def callback_evaluation_summary(self,project_id,*,since_ms=None,until_ms=None):
        """Read reported metric groups; evaluator versions stay separate."""
        query=dict(project_id=self._id(project_id))
        for key,value in [('since_ms',since_ms),('until_ms',until_ms)]:
            if value is not None:
                if type(value) is not int or not 0<=value<=2**64-1:raise ValueError('Invalid evaluation summary time')
                query[key]=value
        if since_ms is not None and until_ms is not None and since_ms>until_ms:raise ValueError('Invalid evaluation summary range')
        return self.request('observability/evaluations/summary?'+urllib.parse.urlencode(query))

    def export_callback_evaluation_summary(self,project_id,*,format='json',since_ms=None,until_ms=None):
        """Export all reported groups from one scoped receipt, without model calls.

        CSV repeats global counters on each row; an empty summary has one context row.
        Evaluator identities and scores are reported by caller code.
        """
        if format not in ('json','csv'):raise ValueError('Invalid callback summary export format')
        packet=self.callback_evaluation_summary(project_id,since_ms=since_ms,until_ms=until_ms)
        def invalid():raise EvaluationError('invalid_callback_summary_export')
        if not isinstance(packet,dict):invalid()
        expected=dict(kind='callback_evaluation_summary',project_id=project_id,since_ms=since_ms,until_ms=until_ms,assessment_source='caller_reported')
        if any(key not in packet or packet[key]!=value for key,value in expected.items()):invalid()
        for key,value in [('since_ms',since_ms),('until_ms',until_ms)]:
            if value is not None and type(packet[key]) is not int:invalid()
        if type(packet.get('provider_calls')) is not int or packet['provider_calls']!=0 or packet.get('automatic_promotion') is not False:invalid()
        counters=['trace_count','selected_tasks','skipped_tasks','completed_assessments','failed_assessments']
        if any(type(packet.get(key)) is not int or not 0<=packet[key]<=20000000 for key in counters):invalid()
        rows=packet.get('metrics')
        if not isinstance(rows,list) or len(rows)>4000:invalid()
        def identifier(value):return isinstance(value,str) and 1<=len(value)<=100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-' for c in value)
        seen=set()
        for row in rows:
            if not isinstance(row,dict) or not identifier(row.get('metric')):invalid()
            if 'evaluator_id' not in row or 'evaluator_version' not in row:invalid()
            identity=row['evaluator_id'];version=row['evaluator_version']
            if not (identity is None and version is None) and not (identifier(identity) and type(version) is int and 1<=version<=2**64-1):invalid()
            if type(row.get('count')) is not int or not 1<=row['count']<=20000000:invalid()
            if any(type(row.get(key)) not in (int,float) or not 0<=row[key]<=1 or not math.isfinite(row[key]) for key in ('mean','min','max')):invalid()
            if not row['min']<=row['mean']<=row['max']:invalid()
            key=(identity,version,row['metric'])
            if key in seen:invalid()
            seen.add(key)
        if 'evaluators' in packet:
            evaluators=packet['evaluators']
            if not isinstance(evaluators,list) or len(evaluators)>4000:invalid()
            identities=set();totals=[0,0]
            for row in evaluators:
                if not isinstance(row,dict) or set(row)!={'evaluator_id','evaluator_version','completed_assessments','failed_assessments'}:invalid()
                identity=row['evaluator_id'];version=row['evaluator_version']
                if not (identity is None and version is None) and not (identifier(identity) and type(version) is int and 1<=version<=2**64-1):invalid()
                key=(identity,version)
                if key in identities:invalid()
                identities.add(key)
                for index,name in enumerate(('completed_assessments','failed_assessments')):
                    if type(row[name]) is not int or not 0<=row[name]<=20000000:invalid()
                    totals[index]+=row[name]
                if row['completed_assessments']+row['failed_assessments']==0:invalid()
            if totals!=[packet['completed_assessments'],packet['failed_assessments']]:invalid()
            if any((row['evaluator_id'],row['evaluator_version']) not in identities for row in rows):invalid()
        if format=='json':return json.dumps(packet,ensure_ascii=False,allow_nan=False,indent=2)
        columns=['project_id','since_ms','until_ms']+counters+['assessment_source','provider_calls','automatic_promotion','evaluator_id','evaluator_version','metric','count','mean','min','max']
        def cell(value):
            if value is None:return ''
            if type(value) is bool:return str(value).lower()
            if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):return "'"+value
            return value
        stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(columns)
        for row in rows or [{}]:writer.writerow([cell(row.get(key) if key in ('evaluator_id','evaluator_version','metric','count','mean','min','max') else packet[key]) for key in columns])
        return stream.getvalue()

    def export_callback_evaluator_attempts_csv(self,project_id,*,since_ms=None,until_ms=None):
        """Export attempt groups, including failed-only and unidentified evaluators."""
        packet=json.loads(self.export_callback_evaluation_summary(project_id,since_ms=since_ms,until_ms=until_ms))
        if 'evaluators' not in packet:raise EvaluationError('callback_evaluator_attempts_unavailable')
        columns=['project_id','since_ms','until_ms','assessment_source','provider_calls','automatic_promotion','evaluator_id','evaluator_version','completed_assessments','failed_assessments']
        stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(columns)
        for row in packet['evaluators'] or [{}]:
            values=[]
            for key in columns:
                value=row.get(key) if key in ('evaluator_id','evaluator_version','completed_assessments','failed_assessments') else packet[key]
                if value is None:value=''
                elif type(value) is bool:value=str(value).lower()
                elif isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):value="'"+value
                values.append(value)
            writer.writerow(values)
        return stream.getvalue()

    def check_callback_evaluation_summary(self,project_id,requirements,*,since_ms=None,until_ms=None,max_failed_assessments=None):
        """Check explicit version-pinned thresholds against one reported summary.

        Missing groups and insufficient samples fail. Failure limit applies to the
        entire scoped summary, since failures may not contain scored metrics.
        This is caller-reported CI evidence, not independent quality certification.
        """
        if not isinstance(requirements,list) or not 1<=len(requirements)<=100:raise ValueError('Use 1..100 callback metric requirements')
        normalized=[];seen=set()
        for requirement in requirements:
            if not isinstance(requirement,dict) or set(requirement)-{'evaluator_id','evaluator_version','metric','min_count','min_mean','min_score','max_failed_assessments'} or not {'evaluator_id','evaluator_version','metric','min_count'}<=set(requirement):raise ValueError('Invalid callback metric requirement')
            for key in ('evaluator_id','metric'):
                value=requirement[key]
                if not isinstance(value,str) or not 1<=len(value)<=100 or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-' for c in value):raise ValueError('Use technical evaluator and metric identifiers')
            if type(requirement['evaluator_version']) is not int or not 1<=requirement['evaluator_version']<=2**64-1:raise ValueError('Pin a positive evaluator version')
            if type(requirement['min_count']) is not int or not 1<=requirement['min_count']<=20000000:raise ValueError('Require a positive bounded sample count')
            if not {'min_mean','min_score'}&set(requirement):raise ValueError('Specify a quality threshold')
            for key in ('min_mean','min_score'):
                if key in requirement and (type(requirement[key]) not in (int,float) or not 0<=requirement[key]<=1 or not math.isfinite(requirement[key])):raise ValueError('Use finite thresholds in 0..1')
            if 'max_failed_assessments' in requirement and (type(requirement['max_failed_assessments']) is not int or not 0<=requirement['max_failed_assessments']<=20000000):raise ValueError('Invalid evaluator failure limit')
            identity=(requirement['evaluator_id'],requirement['evaluator_version'],requirement['metric'])
            if identity in seen:raise ValueError('Duplicate callback metric requirement')
            seen.add(identity);normalized.append(dict(requirement))
        if max_failed_assessments is not None and (type(max_failed_assessments) is not int or not 0<=max_failed_assessments<=20000000):raise ValueError('Invalid callback failure limit')
        summary=json.loads(self.export_callback_evaluation_summary(project_id,since_ms=since_ms,until_ms=until_ms))
        groups={(row['evaluator_id'],row['evaluator_version'],row['metric']):row for row in summary['metrics']}
        attempts={(row['evaluator_id'],row['evaluator_version']):row for row in summary.get('evaluators',[])}
        checks=[]
        for requirement in normalized:
            row=groups.get((requirement['evaluator_id'],requirement['evaluator_version'],requirement['metric']))
            reasons=[]
            if row is None:reasons.append('missing_metric_group')
            else:
                if row['count']<requirement['min_count']:reasons.append('insufficient_samples')
                if 'min_mean' in requirement and row['mean']<requirement['min_mean']:reasons.append('mean_below_threshold')
                if 'min_score' in requirement and row['min']<requirement['min_score']:reasons.append('minimum_below_threshold')
            check=dict(requirement=requirement,observed=row,passed=not reasons,reasons=reasons)
            if 'max_failed_assessments' in requirement:
                attempt=attempts.get((requirement['evaluator_id'],requirement['evaluator_version']))
                check['evaluator_attempts']=attempt
                if attempt is None:reasons.append('evaluator_attempts_unavailable')
                elif attempt['failed_assessments']>requirement['max_failed_assessments']:reasons.append('evaluator_failure_limit_exceeded')
                check['passed']=not reasons
            checks.append(check)
        failure_check=None if max_failed_assessments is None else dict(max_failed_assessments=max_failed_assessments,observed=summary['failed_assessments'],passed=summary['failed_assessments']<=max_failed_assessments,scope='entire_summary')
        return dict(kind='callback_evaluation_gate',passed=all(check['passed'] for check in checks) and (failure_check is None or failure_check['passed']),checks=checks,failure_check=failure_check,summary=summary,assessment_source='caller_reported',provider_calls=0,automatic_promotion=False)

    def trace_time_series(self, *, since_ms, until_ms, bucket_ms, project_id=None, session_id=None, status=None):
        """Bounded reported telemetry assigned by trace start, with empty intervals."""
        if any(type(value) is not int or not 0<=value<=2**64-1 for value in (since_ms,until_ms,bucket_ms)):
            raise ValueError('Use unsigned integer trace series times')
        if since_ms>until_ms or until_ms==2**64-1 or not 1<=bucket_ms<=86400000 or (until_ms-since_ms)//bucket_ms+1>500:
            raise ValueError('Invalid trace series range or interval count')
        query=dict(since_ms=since_ms,until_ms=until_ms,bucket_ms=bucket_ms)
        for name,value in (('project_id',project_id),('session_id',session_id)):
            if value is not None:query[name]=self._id(value)
        if status is not None:
            if not isinstance(status,str) or not 1<=len(status)<=40 or any(char not in 'abcdefghijklmnopqrstuvwxyz_' for char in status):
                raise ValueError('Invalid trace status')
            query['status']=status
        return self.request('observability/time-series?'+urllib.parse.urlencode(query))

    def trace_summary(self, *, project_id=None, session_id=None, since_ms=None, until_ms=None, status=None, conversation_offset=None, conversation_limit=None):
        query={name:value for name,value in dict(project_id=project_id,session_id=session_id,since_ms=since_ms,until_ms=until_ms).items() if value is not None}
        if status is not None:
            if not isinstance(status,str) or not 1<=len(status)<=40 or any(char not in 'abcdefghijklmnopqrstuvwxyz_' for char in status):
                raise ValueError('Invalid trace status')
            query['status']=status
        for name,value,low,high in (('conversation_offset',conversation_offset,0,10000),('conversation_limit',conversation_limit,1,100)):
            if value is not None:
                if type(value) is not int or not low<=value<=high:
                    raise ValueError('Invalid conversation pagination')
                query[name]=value
        return self.request('observability/summary?'+urllib.parse.urlencode(query))

    def datasets(self, project_id, *, archived=False, offset=None, limit=None, search=None):
        if type(archived) is not bool:
            raise ValueError('Use a boolean archive filter')
        query=dict(project_id=self._id(project_id),archived=str(archived).lower())
        for name,value,low,high in (('offset',offset,0,2000),('limit',limit,1,100)):
            if value is not None:
                if type(value) is not int or not low<=value<=high:raise ValueError('Invalid dataset pagination')
                query[name]=value
        if search is not None:
            if not isinstance(search,str) or len(search)>200 or len(search.encode('utf-8'))>800:raise ValueError('Dataset search exceeds 200 characters')
            query['q']=search
        return self.request('evaluation/datasets?'+urllib.parse.urlencode(query))

    def fork_dataset(self,dataset_id,dataset_version,project_id,name):
        dataset_id=self._id(dataset_id);project_id=self._id(project_id)
        if type(dataset_version) is not int or not 1<=dataset_version<=1000:
            raise ValueError('Invalid dataset version')
        if not isinstance(name,str) or not name.strip() or len(name.encode('utf-8'))>200:
            raise ValueError('Invalid dataset name')
        source=self.request('evaluation/datasets/'+dataset_id+'/versions/'+str(dataset_version))
        digest=source.get('sha256')
        if source.get('id')!=dataset_id or source.get('project_id')!=project_id or source.get('version')!=dataset_version or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Dataset source identity/hash mismatch')
        return self.request('evaluation/datasets',dict(project_id=project_id,name=name,base_version=0,samples=source['samples'],origin=dict(id=dataset_id,version=dataset_version,sha256=digest)))

    def save_chat_prompt(self,project_id,name,messages,*,system='Answer the supplied evaluation sample.',prompt_id=None,base_version=0,origin=None):
        project_id=self._id(project_id)
        if type(base_version) is not int or base_version<0 or (prompt_id is None and base_version!=0):raise ValueError('Invalid prompt base version')
        if prompt_id is not None:prompt_id=self._id(prompt_id)
        if not isinstance(name,str) or not name.strip() or len(name.encode('utf-8'))>200 or not isinstance(system,str) or not system.strip() or len(system.encode('utf-8'))>16000:raise ValueError('Invalid prompt name/system')
        messages=json.loads(json.dumps(messages,allow_nan=False))
        if (not isinstance(messages,list) or not 1<=len(messages)<=15 or len(messages)%2==0
                or any(not isinstance(m,dict) or set(m)!=set(('role','content')) or m.get('role')!=('user' if i%2==0 else 'assistant') or not isinstance(m.get('content'),str) or not m['content'].strip() or len(m['content'].encode('utf-8'))>16000 for i,m in enumerate(messages))
                or sum(len(m['content'].encode('utf-8')) for m in messages)>64000 or '{{input}}' not in messages[-1]['content']):
            raise ValueError('Use alternating User/Assistant messages ending with a User question containing {{input}}')
        body=dict(project_id=project_id,name=name,template='',messages=messages,system=system,base_version=base_version)
        if prompt_id is not None:body['id']=prompt_id
        if origin is not None:body['origin']=json.loads(json.dumps(origin,allow_nan=False))
        return self.request('evaluation/prompts',body)

    def start_playground(self,prompt_id,prompt_version,prompt_sha256,input_text,settings,*,contexts=None,expected_output=None,metrics=None,item_timeout_secs=120):
        """Run one question on a reviewed prompt; saves its sample/run and native trace."""
        prompt_id=self._id(prompt_id)
        if type(prompt_version) is not int or prompt_version<1 or not isinstance(prompt_sha256,str) or len(prompt_sha256)!=64 or any(c not in '0123456789abcdef' for c in prompt_sha256):
            raise ValueError('Pin a saved prompt version/hash')
        if not isinstance(input_text,str) or not input_text.strip() or len(input_text.encode('utf-8'))>65536:
            raise ValueError('Invalid playground question')
        contexts=[] if contexts is None else contexts;metrics=[] if metrics is None else metrics
        allowed=('exact_match','contains_reference','json_valid','whitespace_token_f1','character_bigram_f1','json_equals')
        if not isinstance(contexts,list) or len(contexts)>50 or any(not isinstance(c,str) or len(c.encode('utf-8'))>65536 for c in contexts):
            raise ValueError('Invalid playground contexts')
        if expected_output is not None and (not isinstance(expected_output,str) or len(expected_output.encode('utf-8'))>65536):
            raise ValueError('Invalid playground reference')
        if not isinstance(metrics,list) or len(metrics)>6 or any(not isinstance(m,str) or m not in allowed for m in metrics) or len(set(metrics))!=len(metrics):
            raise ValueError('Invalid playground metrics')
        if any(m!='json_valid' for m in metrics) and expected_output is None:
            raise ValueError('Selected metrics require a reference')
        if type(item_timeout_secs) is not int or not 1<=item_timeout_secs<=600:
            raise ValueError('Invalid playground timeout')
        settings=json.loads(json.dumps(settings,allow_nan=False))
        if not isinstance(settings,dict) or settings.get('mode')!='chat' or settings.get('allow_writes',False) is not False:
            raise ValueError('Playground requires Chat without writes')
        self._id(settings.get('project_id'))
        return self.request('evaluation/playground',dict(settings=settings,prompt_ref=dict(id=prompt_id,version=prompt_version),prompt_sha256=prompt_sha256,input=input_text,contexts=list(contexts),expected_output=expected_output,metrics=list(metrics),item_timeout_secs=item_timeout_secs))

    def preview_prompt(self,prompt_id,prompt_version,project_id,input_text,*,contexts=None):
        prompt_id=self._id(prompt_id);project_id=self._id(project_id)
        if type(prompt_version) is not int or prompt_version<1 or not isinstance(input_text,str) or not input_text.strip() or len(input_text.encode('utf-8'))>65536:
            raise ValueError('Invalid prompt preview input/version')
        contexts=[] if contexts is None else contexts
        if not isinstance(contexts,list) or len(contexts)>50 or any(not isinstance(c,str) or len(c.encode('utf-8'))>65536 for c in contexts):
            raise ValueError('Invalid prompt preview contexts')
        return self.request('evaluation/prompts/'+prompt_id+'/versions/'+str(prompt_version)+'/preview',dict(project_id=project_id,input=input_text,contexts=list(contexts)))

    def compare_dataset_versions(self,dataset_id,project_id,from_version,to_version,*,offset=0,limit=100):
        dataset_id=self._id(dataset_id);project_id=self._id(project_id)
        for value,low,high in ((from_version,1,1000),(to_version,1,1000),(offset,0,4000),(limit,1,100)):
            if type(value) is not int or not low<=value<=high:
                raise ValueError('Invalid dataset comparison bounds')
        query=dict(project_id=project_id,from_version=from_version,to_version=to_version,offset=offset,limit=limit)
        return self.request('evaluation/datasets/'+dataset_id+'/compare?'+urllib.parse.urlencode(query))

    def export_dataset_comparison_csv(self,dataset_id,project_id,from_version,to_version):
        """Export every changed sample with pinned revision hashes; no sample text."""
        rows=[];offset=0;pin=None;previous=None;observed=dict(added=0,removed=0,changed=0)
        def invalid():raise EvaluationError('invalid_dataset_comparison',dataset_id)
        while True:
            page=self.compare_dataset_versions(dataset_id,project_id,from_version,to_version,offset=offset,limit=100)
            if not isinstance(page,dict):invalid()
            changes=page.get('changes');total=page.get('total');counts=page.get('counts');before=page.get('from');after=page.get('to')
            if (page.get('id')!=dataset_id or page.get('project_id')!=project_id or page.get('provider_calls')!=0
                    or page.get('offset')!=offset or page.get('limit')!=100 or page.get('order')!='sample_id_asc'
                    or type(total) is not int or not 0<=total<=4000 or not isinstance(changes,list)
                    or len(changes)!=min(100,max(0,total-offset)) or type(page.get('has_more')) is not bool
                    or page['has_more']!=(offset+len(changes)<total) or not isinstance(counts,dict)
                    or any(type(counts.get(k)) is not int or counts[k]<0 for k in ('added','removed','changed','unchanged'))
                    or sum(counts[k] for k in observed)!=total or not isinstance(before,dict) or not isinstance(after,dict)
                    or before.get('version')!=from_version or after.get('version')!=to_version):invalid()
            for snapshot in (before,after):
                digest=snapshot.get('sha256')
                if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):invalid()
            identity=(before['sha256'],after['sha256'],total,tuple(counts[k] for k in ('added','removed','changed','unchanged')))
            if pin is None:pin=identity
            elif pin!=identity:invalid()
            for change in changes:
                if not isinstance(change,dict):invalid()
                sample_id=change.get('sample_id');kind=change.get('kind');fields=change.get('fields')
                if (not isinstance(sample_id,str) or not sample_id or previous is not None and sample_id<=previous
                        or not isinstance(kind,str) or kind not in observed or not isinstance(fields,list) or any(not isinstance(f,str) for f in fields) or len(fields)!=len(set(fields))
                        or any(f not in ('input','expected_output','contexts','metadata') for f in fields)
                        or (kind=='changed')!=bool(fields)):invalid()
                self._id(sample_id);previous=sample_id;observed[kind]+=1
                rows.append([dataset_id,project_id,from_version,before['sha256'],to_version,after['sha256'],sample_id,kind,';'.join(fields)])
            if not page['has_more']:break
            offset+=100
        if any(observed[k]!=counts[k] for k in observed):invalid()
        stream=io.StringIO(newline='');writer=csv.writer(stream)
        writer.writerow(['dataset_id','project_id','from_version','from_sha256','to_version','to_sha256','sample_id','change','fields'])
        writer.writerows([[("'"+v if v.lstrip().startswith(('=','+','-','@')) or v.startswith(('\t','\r','\n')) else v) if isinstance(v,str) else v for v in row] for row in rows])
        return stream.getvalue()

    def dataset_versions(self,dataset_id,project_id,*,offset=0,limit=20):
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Invalid dataset version pagination')
        return self.request('evaluation/datasets/'+self._id(dataset_id)+'/versions?'+urllib.parse.urlencode(dict(project_id=self._id(project_id),offset=offset,limit=limit)))

    def dataset_lifecycle(self, dataset_id, project_id, *, base_version, base_revision, archived):
        if type(archived) is not bool or type(base_version) is not int or not 1<=base_version<=2**64-1 or type(base_revision) is not int or not 0<=base_revision<=2**64-1:
            raise ValueError('Use pinned dataset/lifecycle versions and a boolean archive state')
        return self.request('evaluation/datasets/'+self._id(dataset_id)+'/lifecycle',dict(project_id=self._id(project_id),base_version=base_version,base_revision=base_revision,archived=archived))

    def plan_judges(self, dataset_id, dataset_version, outputs, settings, rubric=None, *, rubric_ref=None, judge_preset=None, experiment_id=None, offline_score_id=None):
        """Persist a prevalidated frozen batch plan; does not invoke a model."""
        if type(dataset_version) is not int or dataset_version<=0:
            raise ValueError("Pin a positive dataset version")
        if sum(value is not None for value in (rubric,rubric_ref,judge_preset))!=1:raise ValueError('Choose one judge criterion source')
        body=dict(dataset_id=self._id(dataset_id),dataset_version=dataset_version,outputs=outputs,settings=settings)
        if judge_preset is not None:
            if not isinstance(judge_preset,dict) or set(judge_preset)!={'id','version'} or type(judge_preset['version']) is not int or judge_preset['version']<=0:raise ValueError('Pin a positive judge preset version')
            body['judge_preset']=dict(id=self._id(judge_preset['id']),version=judge_preset['version'])
        elif rubric_ref is not None:
            if not isinstance(rubric_ref,dict) or set(rubric_ref)!={'id','version'} or type(rubric_ref['version']) is not int or rubric_ref['version']<=0:raise ValueError('Pin a positive rubric version')
            body['rubric_ref']=dict(id=self._id(rubric_ref['id']),version=rubric_ref['version'])
        else:body['rubric']=rubric
        if experiment_id is not None and offline_score_id is not None:raise ValueError('Choose one saved answer source')
        if experiment_id is not None:body['experiment_id']=self._id(experiment_id)
        if offline_score_id is not None:body['offline_score_id']=self._id(offline_score_id)
        return self.request("evaluation/judge-plans",body)

    def plan_experiment_judges(self, run_id, settings, rubric=None, *, rubric_ref=None, judge_preset=None):
        """Freeze completed experiment answers for explicit later rubric judging."""
        run_id=self._id(run_id)
        run=self.run(run_id)
        items=run.get('items',[])
        if run.get('id')!=run_id or run.get('status')!='completed' or not isinstance(items,list) or not 1<=len(items)<=200:
            raise EvaluationError('experiment_not_completed',run_id)
        outputs={}
        for item in items:
            sample_id=item.get('sample_id');output=item.get('output')
            if not isinstance(sample_id,str) or sample_id in outputs or item.get('status')!='completed' or item.get('output_truncated',False) or not isinstance(output,str):
                raise EvaluationError('experiment_output_unavailable',run_id)
            outputs[sample_id]=output
        return self.plan_judges(run['dataset_id'],run['dataset_version'],outputs,settings,
                rubric,rubric_ref=rubric_ref,judge_preset=judge_preset,experiment_id=run_id)

    def plan_scored_output_judges(self, receipt_id, settings, rubric=None, *, rubric_ref=None, judge_preset=None):
        """Freeze verified saved callback/ready answers for explicit later judging."""
        receipt_id=self._id(receipt_id)
        receipt=self.scored_outputs(receipt_id)
        if receipt.get('id')!=receipt_id or receipt.get('kind')!='offline_score' or receipt.get('project_id')!=settings.get('project_id'):
            raise EvaluationError('offline_source_mismatch',receipt_id)
        items=receipt.get('items')
        if not isinstance(items,list) or not 1<=len(items)<=200:
            raise EvaluationError('offline_outputs_unavailable',receipt_id)
        outputs={}
        for item in items:
            sample_id=item.get('sample_id');output=item.get('output')
            if not isinstance(sample_id,str) or sample_id in outputs or not isinstance(output,str):
                raise EvaluationError('offline_outputs_unavailable',receipt_id)
            outputs[sample_id]=output
        return self.plan_judges(receipt['dataset_id'],receipt['dataset_version'],outputs,settings,rubric,
            rubric_ref=rubric_ref,judge_preset=judge_preset,offline_score_id=receipt_id)

    def evaluate_scored_output_judges(self, receipt_id, settings, rubric=None, *,
                                     rubric_ref=None, judge_preset=None, min_scores=None,
                                     min_judge_score=None, baseline_score_id=None,
                                     baseline_judge_id=None, require_improvement=False,
                                     require_judge_improvement=False, timeout=300):
        """Explicit combined deterministic/judge gate over saved answers.

        Both gates must pass. Judging runs even if a completed deterministic gate
        fails, preserving diagnostic evidence. Tasks are never executed again.
        """
        receipt_id=self._id(receipt_id)
        if require_improvement and baseline_score_id is None:
            raise ValueError('Deterministic improvement requires a score baseline')
        if require_judge_improvement and baseline_judge_id is None:
            raise ValueError('Judge improvement requires a judge baseline')
        if min_judge_score is not None and (type(min_judge_score) not in (int,float) or not math.isfinite(min_judge_score) or not 0<=min_judge_score<=1):
            raise ValueError('Invalid judge threshold')
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('Invalid evaluation timeout')
        receipt=self.scored_outputs(receipt_id)
        if receipt.get('id')!=receipt_id or receipt.get('kind')!='offline_score' or receipt.get('project_id')!=settings.get('project_id'):
            raise EvaluationError('offline_source_mismatch',receipt_id)
        thresholds={} if min_scores is None else dict(min_scores)
        means=receipt.get('mean_scores',{})
        for metric,value in thresholds.items():
            if metric not in receipt.get('metrics',[]) or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid deterministic threshold')
        if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in means.values()):
            raise EvaluationError('invalid_offline_scores',receipt_id)
        passed=all(metric in means and means[metric]>=value for metric,value in thresholds.items())
        comparison=None
        if baseline_score_id is not None:
            comparison=self.compare_scored_outputs(self._id(baseline_score_id),receipt_id)
            if comparison.get('kind')!='offline_comparison' or comparison.get('candidate_id')!=receipt_id or comparison.get('baseline_id')!=baseline_score_id or type(comparison.get('regressions')) is not int or comparison['regressions']<0:
                raise EvaluationError('invalid_comparison_receipt',receipt_id)
            passed=passed and comparison['regressions']==0 and (not require_improvement or comparison.get('eligible') is True)
        deterministic=dict(passed=passed,receipt=receipt,thresholds=thresholds,comparison=comparison)
        plan=None
        try:
            plan=self.plan_scored_output_judges(receipt_id,settings,rubric,rubric_ref=rubric_ref,judge_preset=judge_preset)
            judged=self.evaluate_judge_plan(plan['id'],min_score=min_judge_score,baseline_id=baseline_judge_id,
                require_improvement=require_judge_improvement,timeout=timeout)
        except (EvaluationError,ValueError,OSError,KeyboardInterrupt) as error:
            error.deterministic_result=deterministic
            error.judge_plan_id=plan.get('id') if plan else None
            error.judge_run_receipt=None
            error.judge_evidence_unavailable=False
            if getattr(error,'run_id',None):
                try:error.judge_run_receipt=self.judge_run(error.run_id)
                except Exception:error.judge_evidence_unavailable=True
            raise
        return dict(kind='client_combined_evaluation',passed=passed and judged['passed'],
            deterministic=deterministic,judge=judged,judge_plan=plan,automatic_promotion=False)

    def review_scored_output_judges(self, receipt_id, judge_run_id, *, min_scores=None, min_judge_score=None):
        """Recheck saved deterministic and completed judge results using GET only."""
        receipt_id=self._id(receipt_id);judge_run_id=self._id(judge_run_id)
        thresholds={} if min_scores is None else dict(min_scores)
        for value in list(thresholds.values())+[min_judge_score]:
            if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
                raise ValueError('Invalid evaluation threshold')
        receipt=self.scored_outputs(receipt_id)
        run=self.judge_run(judge_run_id)
        if receipt.get('id')!=receipt_id or receipt.get('kind')!='offline_score':
            raise EvaluationError('offline_source_mismatch',receipt_id)
        if run.get('id')!=judge_run_id or run.get('status')!='completed':
            raise EvaluationError('judge_run_not_completed',judge_run_id)
        plan=self.judge_plan(self._id(run.get('plan_id')))
        if (plan.get('kind')!='judge_plan' or plan.get('id')!=run.get('plan_id')
                or not isinstance(plan.get('plan_sha256'),str) or not plan['plan_sha256']
                or run.get('plan_sha256')!=plan['plan_sha256']
                or plan.get('offline_score_source',{}).get('id')!=receipt_id):
            raise EvaluationError('judge_source_mismatch',judge_run_id)
        for field in ('project_id','dataset_id','dataset_version','dataset_sha256'):
            if receipt.get(field) is None or plan.get(field)!=receipt[field] or run.get(field)!=receipt[field]:
                raise EvaluationError('judge_source_mismatch',judge_run_id)
        def outputs(items):
            if not isinstance(items,list) or not 1<=len(items)<=200:raise EvaluationError('invalid_output_coverage')
            result={}
            for item in items:
                sample_id=item.get('sample_id');output=item.get('output')
                if not isinstance(sample_id,str) or sample_id in result or not isinstance(output,str):
                    raise EvaluationError('invalid_output_coverage')
                result[sample_id]=output
            return result
        if outputs(receipt.get('items'))!=outputs(plan.get('samples')):
            raise EvaluationError('judge_source_mismatch',judge_run_id)
        means=receipt.get('mean_scores',{})
        if not isinstance(means,dict) or set(means)!=set(receipt.get('metrics',[])) or not means:
            raise EvaluationError('invalid_offline_scores',receipt_id)
        for value in means.values():
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise EvaluationError('invalid_offline_scores',receipt_id)
        if any(metric not in means or value is None for metric,value in thresholds.items()):
            raise ValueError('Invalid deterministic threshold')
        score=run.get('mean_score')
        if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
            raise EvaluationError('invalid_metric_receipt',judge_run_id)
        deterministic=dict(passed=all(means[metric]>=value for metric,value in thresholds.items()),
                           receipt=receipt,thresholds=thresholds,comparison=None)
        judged=dict(passed=min_judge_score is None or score>=min_judge_score,run=run,
                    min_score=min_judge_score,observed=score,comparison=None,automatic_promotion=False)
        return dict(kind='client_combined_evaluation',passed=deterministic['passed'] and judged['passed'],
                    deterministic=deterministic,judge=judged,judge_plan=plan,
                    provider_calls=0,automatic_promotion=False)

    def save_rubric(self, project_id, name, rubric, *, rubric_id=None, base_version=0):
        """Version a judge criterion using the existing immutable instruction library."""
        body=dict(project_id=project_id,name=name,template='{{input}}',system=rubric,base_version=base_version)
        if rubric_id is not None:body['id']=self._id(rubric_id)
        return self.request('evaluation/prompts',body)

    def judge_plan(self, plan_id):
        return self.request("evaluation/judge-plans/"+self._id(plan_id))

    def judge(self, settings, rubric, input_text, output, reference=None, dataset_ref=None):
        """Explicit model scoring by a rubric; returns a persisted verdict and trace ID."""
        return self.request("evaluation/judge",dict(settings=settings,rubric=rubric,
                            input=input_text,output=output,reference=reference,dataset_ref=dataset_ref),timeout=65)

    def judge_trace(self,trace_id,trace_sha256,settings,rubric,input_text,output,reference=None):
        """Judge explicitly supplied text, pinning completed trace metadata separately."""
        trace_id=self._id(trace_id)
        if not isinstance(trace_sha256,str) or len(trace_sha256)!=64 or any(c not in '0123456789abcdef' for c in trace_sha256):raise ValueError('Invalid trace fingerprint')
        return self.request('evaluation/judge',dict(settings=settings,rubric=rubric,input=input_text,output=output,reference=reference,trace_ref=dict(trace_id=trace_id,trace_sha256=trace_sha256)),timeout=65)

    def judge_sample(self, dataset_id, dataset_version, sample_id, settings, rubric, output):
        """Judge an answer with its pinned dataset question, reference and contexts."""
        dataset_id=self._id(dataset_id)
        if type(dataset_version) is not int or dataset_version<=0:
            raise ValueError("Pin a positive dataset version")
        snapshot=self.request("evaluation/datasets/"+dataset_id+"/versions/"+str(dataset_version))
        sample=next((item for item in snapshot.get("samples",[]) if item.get("id")==sample_id),None)
        if sample is None:raise ValueError("Dataset sample not found")
        return self.judge(settings,rubric,sample["input"],output,sample.get("expected_output"),
                          dict(dataset_id=dataset_id,dataset_version=dataset_version,sample_id=sample_id))

    def judge_outputs(self, dataset_id, dataset_version, outputs, settings, rubric, *, min_score=None):
        """Sequential rubric judging of frozen answers; never retries completed calls."""
        dataset_id=self._id(dataset_id)
        if type(dataset_version) is not int or dataset_version<=0:
            raise ValueError("Pin a positive dataset version")
        if min_score is not None and (type(min_score) not in (int,float) or not math.isfinite(min_score) or not 0<=min_score<=1):
            raise ValueError("Judge minimum must be finite from zero to one")
        if not isinstance(rubric,str) or not rubric.strip() or len(rubric.encode('utf-8'))>16000:
            raise ValueError("Invalid judge rubric")
        # Freeze caller-owned inputs before any provider invocation.
        settings=json.loads(json.dumps(settings,allow_nan=False))
        outputs=json.loads(json.dumps(outputs,allow_nan=False))
        snapshot=self.request("evaluation/datasets/"+dataset_id+"/versions/"+str(dataset_version))
        samples=snapshot.get('samples',[])
        if (snapshot.get('id')!=dataset_id or snapshot.get('version')!=dataset_version
                or not isinstance(samples,list) or not 1<=len(samples)<=200):
            raise EvaluationError('invalid_judge_dataset')
        if settings.get('project_id')!=snapshot.get('project_id') or settings.get('mode')!='chat' or settings.get('allow_writes',False):
            raise ValueError('Judge settings must select the dataset project and Chat without writes')
        if not isinstance(outputs,dict) or set(outputs)!={sample['id'] for sample in samples}:
            raise ValueError('Provide exactly one output for every dataset sample')
        for sample in samples:
            values=[(sample['input'],16000),(outputs[sample['id']],64000)]
            if sample.get('expected_output') is not None:values.append((sample['expected_output'],64000))
            if any(not isinstance(text,str) or len(text.encode('utf-8'))>limit for text,limit in values):
                raise ValueError('Judge sample text exceeds bounds')
            if sum(len(text.encode('utf-8')) for text in sample.get('contexts',[]))>16000:
                raise ValueError('Judge contexts exceed bounds')
        completed=[];receipt_ids=[]
        for sample in samples:
            try:
                created=self.judge(settings,rubric,sample['input'],outputs[sample['id']],sample.get('expected_output'),
                        dict(dataset_id=dataset_id,dataset_version=dataset_version,sample_id=sample['id']))
                receipt_id=self._id(created.get('id'));receipt_ids.append(receipt_id)
                receipt=self.judgment(receipt_id)
                source=receipt.get('dataset_ref',{})
                score=receipt.get('score')
                if (source!=dict(dataset_id=dataset_id,dataset_version=dataset_version,dataset_sha256=snapshot.get('sha256'),sample_id=sample['id'])
                        or receipt.get('rubric')!=rubric or receipt.get('output')!=outputs[sample['id']]
                        or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1):
                    raise EvaluationError('invalid_judge_receipt',receipt_id)
                completed.append(dict(sample_id=sample['id'],receipt=receipt))
            except (EvaluationError,ValueError,KeyboardInterrupt) as error:
                error.receipt_ids=list(receipt_ids)
                error.completed=list(completed)
                error.sample_id=sample['id']
                raise
        mean=sum(item['receipt']['score'] for item in completed)/len(completed)
        return dict(dataset_id=dataset_id,dataset_version=dataset_version,dataset_sha256=snapshot['sha256'],
                    items=completed,mean_score=mean,min_score=min_score,
                    passed=min_score is None or mean>=min_score,automatic_promotion=False)

    def judgment(self, receipt_id):
        """Reopen a hash-verified judge receipt without model invocation."""
        receipt_id=self._id(receipt_id)
        receipt=self.request("evaluation/judge/"+receipt_id)
        if not isinstance(receipt,dict) or receipt.get("id")!=receipt_id or receipt.get("kind")!="llm_judge":
            raise EvaluationError("invalid_judge_receipt")
        return receipt

    def evaluate_task(self, dataset_id, dataset_version, task, metrics, *,
                      min_scores=None, baseline_id=None, require_improvement=False, with_contexts=False, trace=None):
        """Run a synchronous local task once per pinned sample, then score outputs.

        The callback receives input text, plus frozen contexts only when selected;
        never the reference answer or metadata. Its calls can
        have user-defined costs/side effects; no automatic execution retries occur.
        Failures preserve original exception identity and attach partial outputs.
        """
        if not callable(task) or inspect.iscoroutinefunction(task) or inspect.isgeneratorfunction(task) or inspect.isasyncgenfunction(task):
            raise ValueError('Use a synchronous task returning a text answer')
        if type(with_contexts) is not bool:
            raise ValueError('Choose explicitly whether to pass reference contexts')
        dataset_id, samples, metrics, thresholds, project_id = self._task_preflight(dataset_id,dataset_version,
            metrics,min_scores,baseline_id,require_improvement)
        self._validate_task_trace(trace,samples,project_id,task)
        outputs = {}
        for sample in samples:
            try:
                with trace.span('sample.'+sample['id']) if trace is not None else nullcontext():
                    output = task(sample['input'],tuple(sample.get('contexts',[]))) if with_contexts else task(sample['input'])
                    if inspect.iscoroutine(output):
                        output.close()
                    if not isinstance(output,str) or len(output.encode('utf-8')) > 64000:
                        raise ValueError('Task must return a text answer within 64000 UTF-8 bytes')
                    outputs[sample['id']] = output
            except BaseException as error:
                error.completed_outputs = dict(outputs)
                error.sample_id = sample['id']
                error.dataset_id = dataset_id
                error.dataset_version = dataset_version
                raise
        try:
            return self.evaluate_outputs(dataset_id,dataset_version,outputs,metrics,min_scores=thresholds,
                baseline_id=baseline_id,require_improvement=require_improvement)
        except BaseException as error:
            error.completed_outputs=dict(outputs)
            error.dataset_id=dataset_id;error.dataset_version=dataset_version
            error.evaluation_phase='score_delivery'
            raise

    def _validate_task_trace(self, trace, samples, project_id, task=None):
        if trace is None:return
        if not isinstance(trace,_ExternalTrace) or trace.client is not self or trace.project_id != project_id:
            raise ValueError('Use an active trace owned by this Studio client')
        with trace._lock:
            parent = trace._parent.get()
            if trace._closed or parent is None or trace.spans[parent]['status'] != 'running':
                raise ValueError('Enter the trace context before evaluating tasks')
            budget=getattr(task,'_allpaka_trace_span_budget',None)
            extra=budget[1] if isinstance(budget,tuple) and len(budget)==2 and budget[0] is trace else 0
            if type(extra) is not int or extra<0:raise ValueError('Invalid task trace span budget')
            if len(trace.spans)+len(samples)*(1+extra)>200:
                raise ValueError('Task samples exceed remaining trace span capacity')

    def _task_preflight(self, dataset_id, dataset_version, metrics, min_scores, baseline_id, require_improvement):
        dataset_id = self._id(dataset_id)
        if type(dataset_version) is not int or dataset_version <= 0:
            raise ValueError('Pin a positive dataset version')
        metrics = list(metrics)
        catalog = self.metrics()
        definitions = {item['id']: item for item in catalog['metrics']}
        if not metrics or len(metrics) > catalog['max_metrics_per_run'] or len(set(metrics)) != len(metrics) or any(metric not in definitions for metric in metrics):
            raise ValueError('Select unique supported metrics')
        thresholds = {} if min_scores is None else dict(min_scores)
        if any(metric not in metrics or type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1 for metric,value in thresholds.items()):
            raise ValueError('Invalid metric threshold')
        if require_improvement and baseline_id is None:
            raise ValueError('Strict improvement requires a baseline')
        snapshot = self.request('evaluation/datasets/'+dataset_id+'/versions/'+str(dataset_version))
        if snapshot.get('id') != dataset_id or snapshot.get('version') != dataset_version:
            raise EvaluationError('dataset_snapshot_mismatch')
        samples = json.loads(json.dumps(snapshot.get('samples'),allow_nan=False))
        if not isinstance(samples,list) or not 1 <= len(samples) <= 200:
            raise ValueError('Task evaluation supports 1-200 samples')
        if baseline_id is not None:
            baseline = self.scored_outputs(self._id(baseline_id))
            if any(baseline.get(key) != snapshot.get(key.removeprefix('dataset_')) for key in ('dataset_id','dataset_version','dataset_sha256')) or baseline.get('metrics') != metrics:
                raise EvaluationError('offline_baseline_mismatch')
        seen = set()
        for sample in samples:
            sample_id = self._id(sample.get('id'))
            if sample_id in seen or not isinstance(sample.get('input'),str):
                raise EvaluationError('invalid_dataset_sample')
            seen.add(sample_id)
            contexts = sample.get('contexts',[])
            if not isinstance(contexts,list) or len(contexts)>50 or any(not isinstance(text,str) or len(text.encode('utf-8'))>64*1024 for text in contexts):
                raise EvaluationError('invalid_dataset_contexts')
            for metric in metrics:
                requirement = definitions[metric]['reference_requirement']
                reference = sample.get('expected_output')
                if requirement != 'none' and not isinstance(reference,str):
                    raise ValueError('Selected metric requires reference answers')
                if requirement == 'nonempty' and not reference or requirement == 'nonempty_tokens' and not reference.split():
                    raise ValueError('Selected metric requires a nonempty reference')
                if requirement == 'valid_json':
                    def unique(pairs):
                        result = {}
                        for key,value in pairs:
                            if key in result:raise ValueError('Duplicate JSON reference key')
                            result[key] = value
                        return result
                    json.loads(reference,parse_constant=_reject_nonfinite,object_pairs_hook=unique)
        return dataset_id, samples, metrics, thresholds, self._id(snapshot.get('project_id'))

    async def evaluate_task_async(self, dataset_id, dataset_version, task, metrics, *,
                                  min_scores=None, baseline_id=None, require_improvement=False,
                                  item_timeout=60, with_contexts=False, concurrency=1, trace=None):
        """Await a task once per input; Studio HTTP work runs outside the event loop.

        Bounded workers preserve partial results and cancellation boundaries.
        Application calls are not retried; timeouts cancel the current awaitable.
        """
        if not callable(task) or inspect.isgeneratorfunction(task) or inspect.isasyncgenfunction(task):
            raise ValueError('Use an asynchronous task returning a text answer')
        if type(item_timeout) not in (int,float) or not math.isfinite(item_timeout) or not 0 < item_timeout <= 600:
            raise ValueError('Choose an item timeout above zero and at most 600 seconds')
        if type(concurrency) is not int or not 1 <= concurrency <= 8:
            raise ValueError('Choose task concurrency from one to eight')
        if type(with_contexts) is not bool:
            raise ValueError('Choose explicitly whether to pass reference contexts')
        dataset_id, samples, metrics, thresholds, project_id = await asyncio.to_thread(self._task_preflight,
            dataset_id,dataset_version,metrics,min_scores,baseline_id,require_improvement)
        self._validate_task_trace(trace,samples,project_id,task)
        outputs, started, failures = {}, [], []
        remaining = iter(samples)
        stopped = False
        async def worker():
            nonlocal stopped
            while not stopped:
                sample = next(remaining,None)
                if sample is None:return
                started.append(sample['id'])
                try:
                    with trace.span('sample.'+sample['id']) if trace is not None else nullcontext():
                        pending = task(sample['input'],tuple(sample.get('contexts',[]))) if with_contexts else task(sample['input'])
                        if not inspect.isawaitable(pending):
                            raise ValueError('Async task must return an awaitable')
                        output = await asyncio.wait_for(pending,timeout=item_timeout)
                        if inspect.iscoroutine(output):output.close()
                        if not isinstance(output,str) or len(output.encode('utf-8')) > 64000:
                            raise ValueError('Task must return a text answer within 64000 UTF-8 bytes')
                        outputs[sample['id']] = output
                except BaseException as error:
                    stopped = True
                    error.sample_id = sample['id']
                    failures.append(error)
                    raise
        workers = [asyncio.create_task(worker()) for _ in range(min(concurrency,len(samples)))]
        try:
            await asyncio.gather(*workers)
        except BaseException as error:
            stopped = True
            for worker_task in workers:
                if not worker_task.done():worker_task.cancel()
            await asyncio.gather(*workers,return_exceptions=True)
            if not hasattr(error,'sample_id') and failures:
                error.sample_id = failures[0].sample_id
            error.completed_outputs = {sample['id']:outputs[sample['id']] for sample in samples if sample['id'] in outputs}
            error.started_sample_ids = list(started)
            error.dataset_id = dataset_id
            error.dataset_version = dataset_version
            raise
        outputs = {sample['id']:outputs[sample['id']] for sample in samples}
        # This writes one complete receipt. Cancellation during HTTP delivery can
        # leave a committed receipt with a lost response; no retry is attempted.
        try:
            return await asyncio.to_thread(self.evaluate_outputs,dataset_id,dataset_version,outputs,metrics,
                min_scores=thresholds,baseline_id=baseline_id,require_improvement=require_improvement)
        except BaseException as error:
            error.completed_outputs=dict(outputs)
            error.started_sample_ids=list(started)
            error.dataset_id=dataset_id;error.dataset_version=dataset_version
            error.evaluation_phase='score_delivery'
            raise

    def evaluate_outputs(self, dataset_id, dataset_version, outputs, metrics, *,
                         min_scores=None, baseline_id=None, require_improvement=False):
        """Persist and verify supplied outputs, then apply thresholds and paired gates."""
        dataset_id = self._id(dataset_id)
        if type(dataset_version) is not int or dataset_version <= 0:
            raise ValueError("Pin a positive dataset version")
        if require_improvement and baseline_id is None:
            raise ValueError("Strict improvement requires a baseline")
        min_scores = {} if min_scores is None else dict(min_scores)
        for metric, minimum in min_scores.items():
            if metric not in metrics or type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
                raise ValueError("Thresholds require selected metrics and finite scores from zero to one")
        if baseline_id is not None:
            baseline_id = self._id(baseline_id)
            baseline = self.scored_outputs(baseline_id)
            if (baseline.get("dataset_id") != dataset_id or baseline.get("dataset_version") != dataset_version
                    or baseline.get("metrics") != metrics):
                raise EvaluationError("offline_baseline_mismatch")
        created = self.score_outputs(dataset_id, dataset_version, outputs, metrics)
        receipt_id = self._id(created.get("id"))
        try:
            receipt = self.scored_outputs(receipt_id)
            if receipt.get("id") != receipt_id or receipt.get("dataset_id") != dataset_id or receipt.get("dataset_version") != dataset_version or receipt.get("metrics") != metrics:
                raise EvaluationError("invalid_offline_receipt", receipt_id)
            thresholds = {}
            for metric, minimum in min_scores.items():
                score = receipt.get("mean_scores", {}).get(metric)
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise EvaluationError("invalid_metric_receipt", receipt_id)
                thresholds[metric] = dict(minimum=minimum, observed=score, passed=score >= minimum)
            comparison = None
            passed = all(item["passed"] for item in thresholds.values())
            if baseline_id is not None:
                comparison = self.compare_scored_outputs(baseline_id, receipt_id)
                if (comparison.get("baseline_id") != baseline_id or comparison.get("candidate_id") != receipt_id
                        or type(comparison.get("regressions")) is not int or comparison["regressions"] < 0):
                    raise EvaluationError("invalid_comparison_receipt", receipt_id)
                passed = passed and comparison["regressions"] == 0 and (not require_improvement or comparison.get("eligible") is True)
            return dict(passed=passed, receipt=receipt, comparison=comparison, thresholds=thresholds,
                        gate_policy="strict_improvement" if require_improvement else "no_paired_regressions")
        except EvaluationError as error:
            raise EvaluationError(error.reason, receipt_id) from None

    def compare_scored_outputs(self, baseline_id, candidate_id):
        """Compare verified offline receipts; any paired decrease blocks eligibility."""
        return self.request("evaluation/score/compare",dict(baseline_id=self._id(baseline_id),
                            candidate_id=self._id(candidate_id)))

    def list_scored_outputs(self, project_id, *, dataset_id=None, offset=0, limit=20):
        """Browse verified score summaries; never includes answer text."""
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Invalid score catalog pagination')
        query=dict(project_id=self._id(project_id),offset=offset,limit=limit)
        if dataset_id is not None:query['dataset_id']=self._id(dataset_id)
        return self.request('evaluation/score?'+urllib.parse.urlencode(query))

    def scored_outputs(self, receipt_id):
        """Read an offline receipt with server-side dataset and score verification."""
        return self.request("evaluation/score/" + self._id(receipt_id))

    def score_outputs(self, dataset_id, dataset_version, outputs, metrics):
        """Score supplied outputs against a pinned dataset, without invoking a model."""
        if type(dataset_version) is not int or dataset_version <= 0:
            raise ValueError("Pin a positive dataset version")
        return self.request("evaluation/score",dict(dataset_id=self._id(dataset_id),
                            dataset_version=dataset_version,outputs=outputs,metrics=metrics))

    def metrics(self):
        """Discover native metrics and reference requirements without provider calls."""
        return self.request("evaluation/metrics")

    def memory_expiry(self,project_id,*,include_global=False,horizon_days=30):
        """Metadata-only declared expiry review; never writes or infers freshness."""
        if type(include_global) is not bool or type(horizon_days) is not int or not 0<=horizon_days<=365:raise ValueError('Use boolean global scope and horizon 0-365 days')
        return self.request('memory/expiry?'+urllib.parse.urlencode(dict(project_id=self._id(project_id),include_global='true' if include_global else 'false',horizon_days=horizon_days)))

    def memory_proposals(self, session_id):
        """List saved proposal metadata; this does not invoke a model."""
        return self.request("sessions/" + self._id(session_id) + "/memory-proposals")

    def memory_proposal(self, proposal_id):
        proposal_id = self._id(proposal_id)
        receipt = self.request("memory/proposals/" + proposal_id)
        if not isinstance(receipt, dict) or receipt.get("id") != proposal_id or not isinstance(receipt.get("notes"), list):
            raise EvaluationError("invalid_memory_proposal")
        return receipt

    def memory_extraction_source_status(self,proposal_id):
        """Compare the original extraction prefix without a provider call or note mutation."""
        return self.request('memory/proposals/'+self._id(proposal_id)+'/source-status')

    def extract_memory(self, session_id, settings, message_count):
        """Explicit provider call; returns proposals without saving memory notes."""
        if type(message_count) is not int or message_count <= 0:
            raise ValueError("Message boundary must be a positive integer")
        return self.request("sessions/" + self._id(session_id) + "/memory-proposals",
                            dict(settings=settings, message_count=message_count), timeout=65)

    def memory_consolidation_proposals(self,project_id,*,offset=0,limit=20):
        """Browse saved candidate metadata without invoking a provider."""
        project_id=self._id(project_id)
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:raise ValueError('Invalid consolidation proposal page')
        return self.request('memory/consolidation-proposals?'+urllib.parse.urlencode(dict(project_id=project_id,offset=offset,limit=limit)))

    def propose_memory_consolidation(self,settings,sources):
        """Explicit provider call producing one saved candidate without saving notes."""
        if not isinstance(settings,dict) or settings.get('mode')!='chat' or settings.get('allow_writes') is not False:raise ValueError('Use Chat settings without writes')
        self._id(settings.get('project_id'))
        if not isinstance(sources,list) or not 2<=len(sources)<=20:raise ValueError('Use 2..20 pinned sources')
        seen=set();pins=[]
        for source in sources:
            if not isinstance(source,dict) or set(source)!={'id','version','sha256'}:raise ValueError('Invalid memory source pin')
            identity=self._id(source['id']);version=source['version'];fingerprint=source['sha256']
            if identity in seen or type(version) is not int or not 1<=version<=1000 or not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):raise ValueError('Invalid or duplicate source pin')
            seen.add(identity);pins.append(dict(id=identity,version=version,sha256=fingerprint))
        return self.request('memory/consolidation-proposals',dict(settings=dict(settings),sources=pins),timeout=65)

    def memory_consolidation_source_status(self,note_id,version):
        """Read revision status of pinned sources, without modifying notes."""
        note_id=self._id(note_id)
        if type(version) is not int or not 1<=version<=1000:raise ValueError('Invalid memory revision')
        return self.request('memory/notes/'+urllib.parse.quote(note_id,safe='')+'/versions/'+str(version)+'/source-status')

    def consolidate_memory(self,project_id,sources,*,name,content):
        """Save explicitly reviewed consolidation with immutable source pins.

        Native admission checks scope, hashes and unchanged latest revisions.
        Source notes are retained; this does not perform semantic merging.
        """
        project_id=self._id(project_id)
        for value,limit in ((name,200),(content,16000)):
            if not isinstance(value,str) or not value.strip() or len(value.encode('utf-8'))>limit:raise ValueError('Invalid consolidated memory text')
        if not isinstance(sources,list) or not 2<=len(sources)<=20:raise ValueError('Use 2..20 pinned memory sources')
        pins=[];seen=set()
        for source in sources:
            if not isinstance(source,dict) or set(source)!={'id','version','sha256'}:raise ValueError('Invalid memory source pin')
            identity=self._id(source['id']);version=source['version'];fingerprint=source['sha256']
            if identity in seen or type(version) is not int or not 1<=version<=1000 or not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):raise ValueError('Invalid or duplicate memory source pin')
            seen.add(identity);pins.append(dict(id=identity,version=version,sha256=fingerprint))
        return self.request('memory/notes',dict(project_id=project_id,name=name,content=content,base_version=0,consolidation_sources=pins))

    def accept_memory(self, proposal_id, note_index, *, name=None, content=None):
        """Save one explicitly reviewed candidate; optional edits retain its origin."""
        if type(note_index) is not int or note_index < 0:
            raise ValueError("Note index must be a nonnegative integer")
        receipt = self.memory_proposal(proposal_id)
        if note_index >= len(receipt["notes"]):
            raise ValueError("Note index is outside proposal")
        note = receipt["notes"][note_index]
        if not isinstance(note, dict):
            raise EvaluationError("invalid_memory_proposal")
        project_id = self._id(receipt.get("project_id"))
        name = note.get("name") if name is None else name
        content = note.get("content") if content is None else content
        for value, limit in ((name, 200), (content, 16000)):
            if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > limit:
                raise ValueError("Invalid reviewed memory text")
        extra={}
        if receipt.get('kind')=='memory_consolidation_proposal':
            sources=receipt.get('consolidation_sources')
            if not isinstance(sources,list) or not 2<=len(sources)<=20:raise EvaluationError('invalid_consolidation_proposal')
            extra['consolidation_sources']=sources
        return self.request("memory/notes", dict(project_id=project_id, name=name,
                            content=content, base_version=0,
                            proposal_source=dict(proposal_id=proposal_id, note_index=note_index),**extra))

    def evaluate_matrix(self, requests, *, labels=None, timeout=300, poll_interval=0.25,
                        min_scores=None, require_improvement=False, persist=False):
        """Evaluate 2–16 frozen variants sequentially against the first variant.

        Each variant creates a native persistent experiment. This returned matrix
        is a client summary. persist=True additionally saves native paired
        evidence; client thresholds/strict-improvement results remain separate. Failures retain completed
        results and the failing index; no variant is retried or promoted.
        """
        if type(persist) is not bool:raise ValueError('Choose explicitly whether to persist the matrix')
        if not isinstance(requests, (list, tuple)) or not 2 <= len(requests) <= 16:
            raise ValueError("Provide 2–16 experiment requests")
        # Freeze caller-owned nested settings before any network/model operation.
        frozen = json.loads(json.dumps(requests, allow_nan=False))
        labels = list(labels) if labels is not None else ["variant_" + str(i + 1) for i in range(len(frozen))]
        if (len(labels) != len(frozen) or any(not isinstance(v, str) or not v.strip()
                or len(v.encode('utf-8')) > 200 for v in labels) or len(set(labels)) != len(labels)):
            raise ValueError("Provide unique nonblank variant labels up to 200 bytes")
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("Timeout and poll interval must be finite and positive")
        first = frozen[0]
        for request in frozen:
            metrics = request['metrics']
            if (not isinstance(metrics, list) or not metrics or len(metrics) != len(set(metrics))
                    or any(metric not in ('exact_match', 'contains_reference', 'json_valid',
                                          'whitespace_token_f1', 'json_equals', 'character_bigram_f1') for metric in metrics)):
                raise ValueError("Select unique supported matrix metrics")
            for key, default, maximum in (('concurrency', 4, 8), ('item_timeout_secs', 120, 600)):
                value = request.get(key, default)
                if type(value) is not int or not 1 <= value <= maximum:
                    raise ValueError("Invalid matrix execution bound")
            self._id(request['dataset_id'])
            if type(request['dataset_version']) is not int or request['dataset_version'] <= 0:
                raise ValueError("Pin a positive dataset version")
            if any(request[key] != first[key] for key in ('dataset_id', 'dataset_version', 'metrics')):
                raise ValueError("Matrix variants require identical dataset pins and metrics")
            settings = request['settings']
            if (settings['project_id'] != first['settings']['project_id'] or settings.get('mode') != 'chat'
                    or settings.get('allow_writes') is not False):
                raise ValueError("Matrix variants require one project and Chat settings without writes")
            ref = request.get('prompt_ref')
            template = request.get('prompt_template', '')
            if ref is not None:
                self._id(ref['id'])
                if template or type(ref.get('version')) is not int or ref['version'] <= 0:
                    raise ValueError("Pin an exclusive prompt reference")
            elif not isinstance(template, str) or '{{input}}' not in template or len(template.encode('utf-8')) > 16000:
                raise ValueError("Provide a bounded prompt template containing {{input}}")
            for metric, minimum in (min_scores or {}).items():
                if metric not in request['metrics'] or type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
                    raise ValueError("Invalid matrix metric threshold")
        results = []
        baseline_id = None
        for index, request in enumerate(frozen):
            try:
                result = self.evaluate(request, baseline_id=baseline_id, timeout=timeout,
                    poll_interval=poll_interval, min_scores=min_scores,
                    require_improvement=require_improvement if index else False)
                results.append(dict(label=labels[index], result=result))
                if result['run']['status'] != 'completed' or result['run'].get('strict_quality') is not True:
                    raise EvaluationError('matrix_variant_incomplete', result['run'].get('id'))
                if index == 0:
                    baseline_id = self._id(result['run']['id'])
            except (EvaluationError, ValueError, KeyError, TypeError, OSError, KeyboardInterrupt) as error:
                error.matrix_results = results
                error.variant_index = index
                error.variant_label = labels[index]
                raise
        report=dict(kind='client_experiment_matrix', baseline_id=baseline_id, variants=results,
                    automatic_promotion=False)
        if persist:
            try:
                receipt=self.save_experiment_matrix(first['settings']['project_id'],baseline_id,
                    [dict(label=row['label'],run_id=row['result']['run']['id']) for row in results])
                if (receipt.get('kind')!='experiment_matrix' or receipt.get('schema_version')!=1
                        or receipt.get('project_id')!=first['settings']['project_id']
                        or receipt.get('baseline_id')!=baseline_id or receipt.get('automatic_promotion') is not False):
                    raise EvaluationError('invalid_matrix_receipt',baseline_id)
                report['native_matrix']=receipt
            except (EvaluationError,ValueError,KeyError,TypeError,OSError,KeyboardInterrupt) as error:
                error.matrix_results=results;error.matrix_persistence_failed=True
                raise
        return report

    def experiments(self,project_id,*,status=None,provider=None,model=None,dataset_id=None,playground=None,offset=0,limit=20):
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:raise ValueError('Invalid experiment pagination')
        query=dict(project_id=self._id(project_id),offset=offset,limit=limit)
        if status is not None:
            if status not in ('running','completed','failed','cancelled','interrupted'):raise ValueError('Invalid experiment status')
            query['status']=status
        for key,value in (('provider',provider),('dataset_id',dataset_id)):
            if value is not None:query[key]=self._id(value)
        if model is not None:
            if not isinstance(model,str) or len(model)>200 or len(model.encode('utf-8'))>800:raise ValueError('Invalid experiment model search')
            query['model']=model
        if playground is not None:
            if type(playground) is not bool:raise ValueError('Invalid Playground filter')
            query['playground']=str(playground).lower()
        return self.request('evaluation/experiments?'+urllib.parse.urlencode(query))

    def run(self, run_id, timeout=10):
        return self.request("evaluation/experiments/" + self._id(run_id), timeout=timeout)

    def save_experiment_matrix(self, project_id, baseline_id, variants):
        """Persist 2–16 completed runs and native conservative comparisons.

        Variants are dictionaries with label/run_id; baseline must be first.
        The server revalidates outputs, dataset pins and every metric pair.
        """
        if not isinstance(variants,(list,tuple)) or not 2<=len(variants)<=16:
            raise ValueError('Select 2-16 matrix variants')
        rows=[];ids=set();labels=set()
        for variant in variants:
            if not isinstance(variant,dict) or set(variant)!={'label','run_id'}:
                raise ValueError('Each variant needs label and run_id')
            label=variant['label'];run_id=self._id(variant['run_id'])
            if not isinstance(label,str) or not label.strip() or len(label.encode('utf-8'))>200 or label in labels or run_id in ids:
                raise ValueError('Use unique runs and bounded unique labels')
            labels.add(label);ids.add(run_id);rows.append(dict(label=label,run_id=run_id))
        baseline_id=self._id(baseline_id)
        if rows[0]['run_id']!=baseline_id:raise ValueError('Place baseline first')
        return self.request('evaluation/matrices',dict(project_id=self._id(project_id),baseline_id=baseline_id,variants=rows))

    def experiment_matrices(self, project_id, *, offset=0, limit=30):
        """List saved matrix metadata; opening revalidates quality evidence."""
        if type(offset) is not int or not 0<=offset<=2000 or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Invalid matrix page')
        return self.request('evaluation/matrices?'+urllib.parse.urlencode(dict(project_id=self._id(project_id),offset=offset,limit=limit)))

    def experiment_matrix(self, matrix_id):
        """Read a persisted matrix, reverified against all pinned native runs."""
        return self.request('evaluation/matrices/'+self._id(matrix_id))

    def start_matrix_job(self, project_id, variants):
        """Start a server-owned sequence; disconnecting this client does not stop it.

        Variants contain label, pinned dataset sha256 and native experiment request.
        Restart recovery never replays inference automatically.
        """
        project_id=self._id(project_id)
        if not isinstance(variants,(list,tuple)) or not 2<=len(variants)<=16:
            raise ValueError('Select 2-16 matrix variants')
        rows=[];labels=set()
        for variant in variants:
            if not isinstance(variant,dict) or not {'label','sha256','request'}<=set(variant) or set(variant)-{'label','sha256','request','prompt_sha256'}:
                raise ValueError('Each variant needs label, sha256 and request')
            label=variant['label'];digest=variant['sha256'];request=variant['request']
            if not isinstance(label,str) or not label.strip() or len(label.encode('utf-8'))>200 or label in labels:
                raise ValueError('Use unique bounded labels')
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Pin a dataset SHA-256')
            if 'prompt_sha256' in variant:
                prompt_hash=variant['prompt_sha256']
                if not isinstance(prompt_hash,str) or len(prompt_hash)!=64 or any(c not in '0123456789abcdef' for c in prompt_hash):
                    raise ValueError('Pin a prompt SHA-256')
            if not isinstance(request,dict) or not isinstance(request.get('settings'),dict) or request['settings'].get('project_id')!=project_id:
                raise ValueError('Each request must belong to the matrix project')
            labels.add(label);rows.append(variant)
        body=dict(project_id=project_id,variants=rows)
        encoded=json.dumps(body,allow_nan=False,ensure_ascii=False).encode('utf-8')
        if len(encoded)>2*1024*1024:raise ValueError('Matrix job exceeds 2 MiB')
        return self.request('evaluation/matrix-jobs',json.loads(encoded))

    def matrix_job(self, job_id):
        """Read server-owned sequence progress and retained native run IDs."""
        return self.request('evaluation/matrix-jobs/'+self._id(job_id))

    def matrix_jobs(self, project_id, *, offset=0, limit=30):
        if type(offset) is not int or not 0<=offset<=1000 or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Invalid matrix job page')
        return self.request('evaluation/matrix-jobs?'+urllib.parse.urlencode(dict(project_id=self._id(project_id),offset=offset,limit=limit)))

    def cancel_matrix_job(self, job_id):
        """Stop admission and request cancellation of this sequence's active run."""
        return self.request('evaluation/matrix-jobs/'+self._id(job_id)+'/cancel',{})

    def resume_matrix_job(self, job_id):
        """Explicitly continue verified completed variants; never replay an interrupted run."""
        return self.request('evaluation/matrix-jobs/'+self._id(job_id)+'/resume',{})

    def retry_matrix_job(self, job_id):
        """Explicitly create a new job, reusing verified completed variants.

        Failed/interrupted/missing variants make NEW model calls; the original
        job and run IDs remain intact and are linked in the new job metadata.
        """
        return self.request('evaluation/matrix-jobs/'+self._id(job_id)+'/retry',{})

    def session_plan(self, session_id):
        """Read durable milestone IDs, revision and bounded reported checkpoints."""
        return self.request('sessions/'+self._id(session_id)+'/plan')

    def update_session_plan(self, session_id, steps, *, base_revision, allow_reopen=False):
        """Optimistic plan edit; Goal completion requires reported criteria/evidence.

        Explicit allow_reopen permits a human to revise completed milestones.
        It is never enabled by the agent's set_plan tool.
        """
        if type(base_revision) is not int or not 0<=base_revision<=2**64-1 or type(allow_reopen) is not bool:
            raise ValueError('Invalid plan revision or reopen flag')
        if not isinstance(steps,(list,tuple)) or len(steps)>30:raise ValueError('Select at most 30 milestones')
        ids=set()
        for step in steps:
            if not isinstance(step,dict) or not {'title','status'}<=set(step) or set(step)-{'id','title','status','acceptance','evidence'}:
                raise ValueError('Invalid milestone fields')
            if not isinstance(step['title'],str) or not step['title'].strip() or len(step['title'].encode('utf-8'))>500 or step['status'] not in ('pending','in_progress','completed'):
                raise ValueError('Invalid milestone title/status')
            if 'id' in step and step['id']!='':
                step_id=self._id(step['id'])
                if step_id in ids:raise ValueError('Duplicate milestone ID')
                ids.add(step_id)
            for field,maximum in [('acceptance',1000),('evidence',2000)]:
                values=step.get(field,[])
                if not isinstance(values,(list,tuple)) or len(values)>4 or any(not isinstance(value,str) or not value.strip() or len(value.encode('utf-8'))>maximum for value in values):
                    raise ValueError('Invalid bounded milestone '+field)
        payload=dict(steps=steps,base_revision=base_revision,allow_reopen=allow_reopen)
        encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
        if len(json.dumps(steps,ensure_ascii=False,separators=(',',':')).encode('utf-8'))>64*1024:raise ValueError('Plan exceeds 64 KiB')
        return self.request('sessions/'+self._id(session_id)+'/plan',json.loads(encoded))

    def comparisons(self, project_id, *, offset=0, limit=30):
        """List metadata summaries; open a receipt to verify its quality result."""
        if type(offset) is not int or not 0<=offset<=2000 or type(limit) is not int or not 1<=limit<=100:
            raise ValueError("Invalid comparison page")
        return self.request("evaluation/comparisons?"+urllib.parse.urlencode(dict(project_id=self._id(project_id),offset=offset,limit=limit)))

    def comparison(self, comparison_id):
        """Read a saved comparison, reverified against pinned native results."""
        return self.request("evaluation/comparisons/"+self._id(comparison_id))

    def export_comparison_csv(self, comparison_id):
        """Export all reverified metric pairs and conservative gate evidence."""
        packet=self.comparison(comparison_id)
        pairs=packet.get('pairs')
        if (packet.get('id')!=comparison_id or packet.get('observational_only') is not True
                or not isinstance(pairs,list) or not 1<=len(pairs)<=1200):
            raise EvaluationError('invalid_comparison_receipt',comparison_id)
        def cell(value):
            if not isinstance(value,str):raise EvaluationError('invalid_comparison_receipt',comparison_id)
            return "'"+value if value.lstrip().startswith(('=','+','-','@')) or value.startswith(('\t','\r','\n')) else value
        fixed=[cell(packet.get(key)) for key in ['id','baseline_id','candidate_id','project_id','dataset_id','dataset_sha256']]
        version=packet.get('dataset_version')
        if type(version) is not int or version<1:raise EvaluationError('invalid_comparison_receipt',comparison_id)
        rows=[];seen=set();regressions=improvements=0
        for pair in pairs:
            if not isinstance(pair,dict):raise EvaluationError('invalid_comparison_receipt',comparison_id)
            sample_id,metric=pair.get('sample_id'),pair.get('metric')
            identity=(cell(sample_id),cell(metric))
            if identity in seen:raise EvaluationError('invalid_comparison_receipt',comparison_id)
            seen.add(identity)
            base,candidate,delta=(pair.get(key) for key in ['baseline','candidate','delta'])
            if (any(type(value) not in (int,float) or not math.isfinite(value) for value in [base,candidate,delta])
                    or not 0<=base<=1 or not 0<=candidate<=1 or delta!=candidate-base):
                raise EvaluationError('invalid_comparison_receipt',comparison_id)
            regressions+=delta<0;improvements+=delta>0
            rows.append([*identity,base,candidate,delta,'regression' if delta<0 else 'improvement' if delta>0 else 'unchanged'])
        eligible=regressions==0 and improvements>0
        reason='paired_regression' if regressions else 'no_improvement' if not improvements else 'strict_paired_improvement'
        if (type(packet.get('regressions')) is not int or packet['regressions']!=regressions
                or type(packet.get('improvements')) is not int or packet['improvements']!=improvements
                or packet.get('eligible') is not eligible or packet.get('reason')!=reason):
            raise EvaluationError('invalid_comparison_receipt',comparison_id)
        stream=io.StringIO(newline='');writer=csv.writer(stream)
        writer.writerow(['comparison_id','baseline_id','candidate_id','project_id','dataset_id','dataset_sha256',
                         'dataset_version','eligible','regressions','improvements','reason','sample_id','metric','baseline','candidate','delta','change'])
        for row in rows:writer.writerow([*fixed,version,eligible,regressions,improvements,reason,*row])
        return stream.getvalue()

    def export_experiment(self, run_id, *, include_outputs=False):
        """Export per-example status/scores; model answers require explicit opt-in.

        Includes incomplete runs without treating them as quality-qualified.
        Prompts, settings, raw usage and error text are omitted.
        """
        if type(include_outputs) is not bool:
            raise ValueError('Choose explicitly whether to export model answers')
        return self.request('evaluation/experiments/'+self._id(run_id)+'/export?'+
                            urllib.parse.urlencode({'include_outputs':str(include_outputs).lower()}))

    def export_experiment_csv(self, run_id, *, include_outputs=False):
        """Tabular snapshot of every example, including failed/pending examples.

        Text cells that spreadsheet programs interpret as formulas are prefixed
        with an apostrophe. Use the JSON export to retain exact answer strings.
        """
        packet=self.export_experiment(run_id,include_outputs=include_outputs)
        if (packet.get('kind')!='experiment_export' or packet.get('schema_version')!=1
                or packet.get('run_id')!=run_id or packet.get('outputs_included') is not include_outputs):
            raise EvaluationError('invalid_experiment_export',run_id)
        metrics=packet.get('metrics')
        items=packet.get('items')
        allowed={'exact_match','contains_reference','json_valid','whitespace_token_f1','character_bigram_f1','json_equals'}
        if (not isinstance(metrics,list) or (not metrics and (packet.get('playground') is not True or packet.get('strict_quality') is not False)) or len(metrics)>6
                or any(not isinstance(metric,str) or metric not in allowed for metric in metrics)
                or len(set(metrics))!=len(metrics) or not isinstance(items,list) or len(items)>200):
            raise EvaluationError('invalid_experiment_export',run_id)
        if type(packet.get('dataset_version')) is not int or packet['dataset_version']<1 or type(packet.get('strict_quality')) is not bool:
            raise EvaluationError('invalid_experiment_export',run_id)
        def text_cell(value):
            if not isinstance(value,str):raise EvaluationError('invalid_experiment_export',run_id)
            return "'"+value if value.lstrip().startswith(('=','+','-','@')) or value.startswith(('\t','\r','\n')) else value
        prompt=packet.get('prompt_ref')
        if prompt is not None and (not isinstance(prompt,dict) or type(prompt.get('version')) is not int or prompt['version']<1):
            raise EvaluationError('invalid_experiment_export',run_id)
        def optional_text(value):return '' if value is None else text_cell(value)
        provenance=[optional_text(packet.get('provider')),optional_text(packet.get('model')),optional_text(packet.get('trace_id')),
            optional_text(prompt.get('id')) if prompt else '',prompt['version'] if prompt else '',optional_text(prompt.get('sha256')) if prompt else '']
        stream=io.StringIO(newline='');writer=csv.writer(stream)
        writer.writerow(['run_id','dataset_id','dataset_version','dataset_sha256','run_status','strict_quality',
                         'sample_id','sample_status','duration_ms','has_error','output_truncated',
                         'provider','model','trace_id','prompt_id','prompt_version','prompt_sha256']+
                        ['score_'+metric for metric in metrics]+(['output'] if include_outputs else []))
        for item in items:
            if not isinstance(item,dict) or not isinstance(item.get('scores'),dict):
                raise EvaluationError('invalid_experiment_export',run_id)
            if type(item.get('has_error')) is not bool or type(item.get('output_truncated')) is not bool:
                raise EvaluationError('invalid_experiment_export',run_id)
            duration=item.get('duration_ms')
            if duration is not None and (type(duration) is not int or duration<0):
                raise EvaluationError('invalid_experiment_export',run_id)
            scores=[]
            for metric in metrics:
                value=item['scores'].get(metric)
                if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
                    raise EvaluationError('invalid_experiment_export',run_id)
                scores.append('' if value is None else value)
            row=[text_cell(run_id),text_cell(packet.get('dataset_id')),packet.get('dataset_version'),
                 text_cell(packet.get('dataset_sha256')),text_cell(packet.get('status')),packet.get('strict_quality'),
                 text_cell(item.get('sample_id')),text_cell(item.get('status')),'' if duration is None else duration,
                 item.get('has_error'),item.get('output_truncated')]+provenance+scores
            if include_outputs:row.append('' if item.get('output') is None else text_cell(item['output']))
            writer.writerow(row)
        return stream.getvalue()

    def cancel(self, run_id):
        return self.request("evaluation/experiments/" + self._id(run_id) + "/cancel", {})

    def wait(self, run_id, timeout=300, poll_interval=0.25):
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("Timeout and poll interval must be finite and positive")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EvaluationError("evaluation_timeout", run_id)
            try:run = self.run(run_id, timeout=min(10, remaining))
            except EvaluationError as error:
                if error.reason=='connection_error' and time.monotonic()>=deadline:
                    raise EvaluationError('evaluation_timeout',run_id) from None
                raise
            if run.get("id") != run_id or run.get("status") not in ("running", "completed", "failed", "cancelled", "interrupted"):
                raise EvaluationError("invalid_run_receipt", run_id)
            if run["status"] != "running":
                return run
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def evaluate(self, request, baseline_id=None, timeout=300, poll_interval=0.25, require_improvement=False, min_scores=None):
        """Run a pinned native evaluation; cancel this new run if waiting fails.

        A baseline always requires zero paired declines. Set require_improvement
        to additionally require a gain. The server verifies receipts and hashes.
        No model settings are promoted automatically.
        """
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("Timeout and poll interval must be finite and positive")
        if require_improvement and baseline_id is None:
            raise ValueError("Strict improvement requires a baseline")
        min_scores = dict(min_scores or {})
        for metric, minimum in min_scores.items():
            if metric not in request["metrics"] or type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
                raise ValueError("Thresholds must name selected metrics and have finite values in [0,1]")
        if baseline_id is not None:
            self._id(baseline_id)
            baseline = self.run(baseline_id)
            if baseline.get("id") != baseline_id or baseline.get("status") != "completed" or baseline.get("strict_quality") is not True:
                raise EvaluationError("invalid_baseline", baseline_id)
            for key in ("dataset_id", "dataset_version", "metrics"):
                if (sorted(baseline.get(key, [])) if key == "metrics" else baseline.get(key)) != (sorted(request[key]) if key == "metrics" else request[key]):
                    raise EvaluationError("baseline_snapshot_mismatch", baseline_id)
            if baseline.get("project_id") != request["settings"]["project_id"]:
                raise EvaluationError("baseline_project_mismatch", baseline_id)
        started = self.request("evaluation/experiments", request)
        run_id = self._id(started.get("id"))
        try:
            run = self.wait(run_id, timeout, poll_interval)
        except (EvaluationError, KeyboardInterrupt) as error:
            cancellation_failed = False
            try:
                self.cancel(run_id)
            except (EvaluationError, ValueError):
                cancellation_failed = True
            if isinstance(error, KeyboardInterrupt):
                error.run_id = run_id
                error.cancellation_failed = cancellation_failed
                raise
            raise EvaluationError(error.reason, run_id, cancellation_failed) from None
        comparison = None
        passed = run["status"] == "completed" and run.get("strict_quality") is True
        if passed and baseline_id is not None:
            try:
                comparison = self.request("evaluation/compare", {"baseline_id": baseline_id, "candidate_id": run_id})
            except EvaluationError as error:
                raise EvaluationError(error.reason, run_id) from None
            if comparison.get("baseline_id") != baseline_id or comparison.get("candidate_id") != run_id:
                raise EvaluationError("invalid_comparison_receipt", run_id)
            if type(comparison.get("regressions")) is not int or comparison["regressions"] < 0:
                raise EvaluationError("invalid_comparison_receipt", run_id)
            passed = comparison["regressions"] == 0 and (not require_improvement or comparison.get("eligible") is True)
        thresholds = {}
        if run["status"] == "completed" and run.get("strict_quality") is True:
            for metric, minimum in min_scores.items():
                score = run.get("mean_scores", {}).get(metric)
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise EvaluationError("invalid_metric_receipt", run_id)
                thresholds[metric] = {"minimum": minimum, "observed": score, "passed": score >= minimum}
            passed = passed and all(row["passed"] for row in thresholds.values())
        return {"run": run, "comparison": comparison, "passed": passed, "thresholds": thresholds, "gate_policy": "strict_improvement" if require_improvement else "no_paired_regression"}
