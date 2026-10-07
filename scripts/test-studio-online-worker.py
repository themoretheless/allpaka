#!/usr/bin/env python3
"""Automatic online evaluation lifecycle over HTTP; no inference provider required."""
import json, os, pathlib, socket, subprocess, tempfile, time, urllib.request

repo=pathlib.Path(__file__).resolve().parents[1]
binary=pathlib.Path(os.environ.get('ALLPAKA_TEST_BINARY',str(repo/'target/debug/allpaka')))
with tempfile.TemporaryDirectory(prefix='allpaka-online-worker-') as tmp:
    root=pathlib.Path(tmp);data=root/'history';data.mkdir();workspace=root/'workspace';workspace.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}/api/'
    env=dict(os.environ,OPENROUTER_API_KEY='',ALLPAKA_LOCAL_BASE_URL='http://127.0.0.1:1/v1')
    env.pop('ALLPAKA_ONLINE_WORKER',None)
    def api(path,body=None):
        request=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Allpaka-Client':'studio'})
        with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)
    log=open(root/'server.log','w');process=None
    def stop():
        global process
        if process is not None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            process=None
    def start(manual=False):
        global process
        child_env=dict(env)
        if manual:child_env['ALLPAKA_ONLINE_WORKER']='0'
        process=subprocess.Popen([str(binary),'studio','--bind',f'127.0.0.1:{port}','--workspace',str(workspace),'--data-dir',str(data)],env=child_env,stdout=log,stderr=log)
        for _ in range(200):
            if process.poll() is not None:raise AssertionError((root/'server.log').read_text())
            try:api('config');return
            except Exception:time.sleep(.03)
        raise AssertionError('Studio startup timed out')
    def ingest(key,status='completed'):
        return api('observability/external-traces',dict(project_id='default',correlation_id=key,idempotency_key=key,started_ms=1000,spans=[dict(parent_id=None,kind='agent',name='external-root',status=status,started_ms=1000,duration_ms=1)]))
    def result_for(trace):
        jobs=data/'observability/online-rules/jobs'
        for path in jobs.glob('*.json'):
            job=json.loads(path.read_text())
            if job['trace_id']==trace:
                result=data/'observability/online-rules/job-results'/path.name
                if result.exists():return result
    def wait_result(trace):
        until=time.monotonic()+10
        while time.monotonic()<until:
            path=result_for(trace)
            if path:return path
            time.sleep(.05)
        raise AssertionError('Automatic evaluation timed out')
    try:
        start(manual=True)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/',timeout=5) as response:html=response.read().decode();assert 'online-jobs-open' in html and 'online-rules-open' in html
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/app.js',timeout=5) as response:script=response.read().decode();assert 'loadOnlineJobs' in script and 'loadOnlineRules' in script
        snapshot=api('observability/online-rules',dict(id='health',project_id='default',evaluator_id='trace_health',evaluator_version=1,sample_rate=1.0,enabled=True))
        api('observability/online-rule-bindings',dict(rule_sha256=snapshot['rule_sha256'],base_version=0,active=True))
        retained=ingest('retained');assert result_for(retained['id']) is None
        pending=api('observability/online-jobs?project_id=default&limit=1');assert pending['total']==1 and pending['jobs'][0]['status']=='pending'
        assert not (data/'observability/online-rules/job-results').exists()
        # Queue exhaustion must preserve successful external ingestion.
        jobs=data/'observability/online-rules/jobs';quota=[]
        for index in range(999):
            path=jobs/f'quota-{index}.json';path.write_text('{}');quota.append(path)
        overflow=ingest('queue-full');assert api('observability/traces/'+overflow['id'])['status']=='completed'
        assert not any(json.loads(path.read_text()).get('trace_id')==overflow['id'] for path in jobs.glob('*.json'))
        for path in quota:path.unlink()
        stop();start()
        retained_path=wait_result(retained['id']);retained_bytes=retained_path.read_bytes()
        result=json.loads(retained_bytes);assert result['status']=='completed' and result['automatic_execution'] is True and result['provider_calls']==0
        fresh=ingest('fresh');fresh_path=wait_result(fresh['id']);fresh_bytes=fresh_path.read_bytes()
        assert json.loads(fresh_bytes)['automatic_execution'] is True
        retry=ingest('fresh');assert retry['id']==fresh['id'] and retry['deduplicated'] is True
        failed=ingest('failed','failed');assert result_for(failed['id']) is None
        assert len(list((data/'observability/online-rules/jobs').glob('*.json')))==2
        assert api('observability/online-jobs/drain',{'limit':20})['already_finished']==2
        first=api('observability/online-jobs?project_id=default&limit=1');second=api('observability/online-jobs?project_id=default&offset=1&limit=1')
        assert first['total']==2 and first['has_more'] and not second['has_more']
        assert first['jobs'][0]['job']['selection_sha256']!=second['jobs'][0]['job']['selection_sha256']
        assert all(page['jobs'][0]['status']=='completed' and page['jobs'][0]['result']['automatic_execution'] is True for page in [first,second])
        pinned_job=first['jobs'][0]['job'];source_input=dict(project_id='default',trace_id=pinned_job['trace_id'],trace_sha256=pinned_job['trace_sha256'],input='Explicit quality question',output='Explicit quality answer',reference=None)
        source=api('observability/online-quality-sources',source_input)
        assert source['source']==source_input and source['answer_source']=='caller_supplied' and source['trace_content_verified'] is False
        assert api('observability/online-quality-sources',source_input)==source
        source_path='observability/online-quality-sources/'+source['source_sha256']+'?project_id=default'
        assert api(source_path)==source
        source_file=data/'evaluation/online-sources'/(source['source_sha256']+'.json');source_bytes=source_file.read_bytes()
        corrupt=dict(source);corrupt['source']=dict(source_input,output='changed');source_file.write_text(json.dumps(corrupt))
        try:api(source_path);raise AssertionError('corrupt source accepted')
        except urllib.error.HTTPError as error:assert error.code==404
        source_file.write_bytes(source_bytes)

        stop();start();time.sleep(2.2)
        assert retained_path.read_bytes()==retained_bytes and fresh_path.read_bytes()==fresh_bytes
        assert len(list((data/'observability/online-rules/job-results').glob('*.json')))==2
        assert api(source_path)==source and source_file.read_bytes()==source_bytes
        print('PASS automatic HTTP worker: retained startup jobs, external admission, deduplication, failed exclusion, queue-capacity isolation, explicit quality-source persistence/tamper rejection, restart without replay; no provider configured')
    finally:stop();log.close()
