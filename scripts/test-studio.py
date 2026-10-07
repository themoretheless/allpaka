#!/usr/bin/env python3
"""Studio end-to-end contract tests against a local streaming provider; no paid API calls."""
import csv, io, json, os, pathlib, socket, subprocess, tempfile, threading, time, urllib.request, urllib.error, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"sdk/python"))
from allpaka_studio import Studio, EvaluationError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

requests = []
guard_stream_started=threading.Event()
class Mock(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        if self.path.startswith('/doctor-'):
            kind=self.path.split('/')[1]
            status={'doctor-auth':401,'doctor-limit':429,'doctor-redirect':302,'doctor-missing':404}.get(kind,200)
            self.send_response(status)
            if status==302:self.send_header('Location','http://127.0.0.1:1/never-follow')
            self.end_headers()
            body=b'{"data":[{"id":"x","id":"y"}]}' if kind=='doctor-invalid' else b'x'*(1024*1024+1) if kind=='doctor-large' else b'{"data":[]}' if kind=='doctor-empty' else b'PROVIDER_SECRET_BODY'
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
            return
        self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"mock","name":"Mock model","context_length":32000,"architecture":{"input_modalities":["text","image"]},"pricing":{"prompt":"0.000001","completion":"0.000002"},"supported_parameters":["tools"]}]}')
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(body)
        messages=body['messages'];last=messages[-1];content=last.get('content','')
        if isinstance(content,list):content=''.join(p.get('text','') for p in content)
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        def event(value):self.wfile.write(('data: '+json.dumps(value)+'\n\n').encode());self.wfile.flush()
        try:
            if content=='GUARD_STREAM_TEST':
                event({'choices':[{'delta':{'content':'BLOCKED_STREAM_TEXT'}}]});guard_stream_started.set();time.sleep(1.0)
                event({'choices':[{'delta':{'content':' tail'},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif any(message.get('content')=='GOAL_STUCK_FIXTURE' for message in messages):
                event({'choices':[{'delta':{'content':'I claim success without a plan.'},'finish_reason':'stop'}]})
            elif any(message.get('content')=='GOAL_CONTINUE_FIXTURE' for message in messages):
                if content=='GOAL_CONTINUE_FIXTURE':
                    args=dict(base_revision=0,steps=[dict(title='Closed fixture',status='completed',acceptance=['First check'],evidence=['Reported first check']),dict(title='Remaining fixture',status='pending',acceptance=['Second check'])])
                    event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'continue-plan-1','function':{'name':'set_plan','arguments':json.dumps(args)}}]},'finish_reason':'tool_calls'}]})
                elif last.get('role')=='user' and content.startswith('Goal continuation:'):
                    system=messages[0]['content'];steps=json.loads(system.rsplit('Current milestones: ',1)[1]);revision=int(system.rsplit('Durable milestone plan revision: ',1)[1].split('.')[0])
                    assert steps[0]['status']=='completed' and steps[1]['status']=='pending'
                    steps[1]['status']='completed';steps[1]['evidence']=['Reported second check']
                    event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'continue-plan-2','function':{'name':'set_plan','arguments':json.dumps(dict(base_revision=revision,steps=steps))}}]},'finish_reason':'tool_calls'}]})
                else:event({'choices':[{'delta':{'content':'Claimed complete.'},'finish_reason':'stop'}]})
            elif 'Goal mode' in messages[0].get('content','') and any(message.get('content')=='DURABLE_MILESTONE_GOAL' for message in messages):
                if content=='DURABLE_MILESTONE_GOAL':
                    args=dict(base_revision=0,steps=[dict(title='Completed milestone',status='completed',acceptance=['Fixture criterion'],evidence=['Reported fixture evidence']),dict(title='Remaining milestone',status='in_progress',acceptance=['Finish after explicit resume'],evidence=[])])
                    event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'goal-plan-1','function':{'name':'set_plan','arguments':json.dumps(args)}}]},'finish_reason':'tool_calls'}]})
                elif last.get('role')=='tool' and json.loads(content).get('revision')==1:
                    event({'choices':[{'delta':{'content':'unfinished milestone'}}]});time.sleep(3)
                    event({'choices':[{'delta':{'content':' later'},'finish_reason':'stop'}]})
                elif last.get('role')=='user':
                    system=messages[0]['content'];steps=json.loads(system.rsplit('Current milestones: ',1)[1]);revision=int(system.rsplit('Durable milestone plan revision: ',1)[1].split('.')[0])
                    assert steps[0]['status']=='completed'
                    steps[1]['status']='completed';steps[1]['evidence']=['Reported completion after explicit resume']
                    event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'goal-plan-2','function':{'name':'set_plan','arguments':json.dumps(dict(base_revision=revision,steps=steps))}}]},'finish_reason':'tool_calls'}]})
                else:event({'choices':[{'delta':{'content':'Milestones reported complete.'},'finish_reason':'stop'}]})
            elif messages[0].get('role')=='system' and messages[0].get('content','').startswith('Summarize conversation history'):
                if 'RETRY_COMPACT' in content: time.sleep(1.2)
                if 'FAIL_COMPACT' in content or ('RETRY_COMPACT' in content and body['max_tokens'] < 32768):
                    event({'choices':[{'delta':{'content':'incomplete'},'finish_reason':'length'}]})
                else:
                    event({'choices':[{'delta':{'content':'COMPACTED: Preserve user goals and completed work.'},'finish_reason':'stop'}]})
            elif messages[0].get('content','')=='Answer the supplied evaluation sample.' and content.startswith('PAIR_TEST '):
                model=body['model'];correct=model=='better' or model=='baseline' and content.endswith('a') or model=='candidate' and not content.endswith('a')
                event({'choices':[{'delta':{'content':'ok' if correct else 'wrong'},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif 'ESCAPED_OUTPUT_FIXTURE' in content:
                event({'choices':[{'delta':{'content':'\x00'*65536},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif content=='JSON_DUPLICATE_TEST':
                event({'choices':[{'delta':{'content':'{"a":1,"a":2}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif content=='JSON_EQUAL_TEST':
                event({'choices':[{'delta':{'content':'{ "b": [true, null], "a": 1 }'},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif messages[0].get('content','').startswith('Evaluate the supplied answer using'):
                if 'SLOW_JUDGE' in content:time.sleep(1.2)
                verdict={'score':0.75,'reason':'The answer satisfies most of the supplied rubric.'}
                if 'JUDGE_SCORE_HIGH' in content:verdict['score']=0.9
                if 'JUDGE_SCORE_LOW' in content:verdict['score']=0.7
                if 'BAD_JUDGE' in content:verdict['score']=2
                event({'choices':[{'delta':{'content':json.dumps(verdict)},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif messages[0].get('content','').startswith('Consolidate the supplied memory notes'):
                value={'notes':[{'name':'Consolidated candidate','content':'Combined fact with explicitly retained qualification.'}]}
                if 'BAD_CONSOLIDATE' in content:value={'notes':[]}
                event({'choices':[{'delta':{'content':json.dumps(value)},'finish_reason':'stop'}],'usage':{'prompt_tokens':11,'completion_tokens':7}})
            elif messages[0].get('content','').startswith('Extract durable decisions'):
                value={'notes':[{'name':'Extracted decision','content':'A proposed fact requiring review.'}]}
                if 'BAD_EXTRACT' in content:value={'notes':[{'name':'','content':'invalid'}]}
                event({'choices':[{'delta':{'content':json.dumps(value)},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif content == 'TRY_MEMORY_RECALL':
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'memory1','type':'function','function':{'name':'memory_recall','arguments':json.dumps({'query':'MEMORY_FIXTURE','limit':5})}}]},'finish_reason':'tool_calls'}]})
            elif content == 'TRY_AGENTGREP':
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'search1','type':'function','function':{'name':'agentgrep','arguments':json.dumps({'path':'workspace','query':'verified edit'})}}]},'finish_reason':'tool_calls'}]})
            elif content == 'TRY_CONVERSATION_SEARCH':
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'history1','type':'function','function':{'name':'conversation_search','arguments':json.dumps({'query':'long turn 0','limit':5})}}]},'finish_reason':'tool_calls'}]})
            elif content == 'TRY_BACKGROUND':
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'bg1','type':'function','function':{'name':'background','arguments':json.dumps({'action':'start','command':'printf background-done','name':'Fixture background'})}}]},'finish_reason':'tool_calls'}]})
            elif 'STREAM_ANALYSIS' in content:
                event({'choices':[{'delta':{'reasoning_content':'Visible analysis <script>literal</script>'}}]})
                time.sleep(1)
                event({'choices':[{'delta':{'content':'Finished answer'},'finish_reason':'stop'}]})
            elif 'TOKEN_LIMIT_TOOL' in content:
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'truncated','function':{'name':'write_file','arguments':'{"path":'}}]},'finish_reason':'length'}],'usage':{'completion_tokens':4096}})
            elif 'TOKEN_LIMIT_TEXT' in content:
                time.sleep(.15)
                event({'choices':[{'delta':{'content':'partial text'},'finish_reason':'length'}],'usage':{'completion_tokens':4096}})
            elif body.get('model')=='matrix-retry-crash' and content.startswith('SLOW_MATRIX_RETRY'):
                event({'choices':[{'delta':{'content':'started '}}]});time.sleep(3)
                event({'choices':[{'delta':{'content':'retry finished'},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            elif 'SLOW' in content:
                event({'choices':[{'delta':{'content':'started '}}]})
                for _ in range(15):time.sleep(.1);event({'choices':[{'delta':{'content':'.'}}]})
            elif any(message.get('role')=='user' and 'NO_PROGRESS_LOOP' in str(message.get('content','')) for message in messages):
                if any(message.get('role')=='tool' and 'loop recovered' in message.get('content','') for message in messages):
                    event({'choices':[{'delta':{'content':'Recovered after explicit resume.'},'finish_reason':'stop'}]})
                else:
                    failed=sum(message.get('role')=='tool' and 'error' in message.get('content','') for message in messages)
                    calls=[{'index':index,'id':'loop'+str(index),'type':'function','function':{'name':'read_file','arguments':json.dumps({'path':'workspace/recovery-loop.txt'})}} for index in range(1 if failed>=3 else 3)]
                    if failed<3:calls.append({'index':3,'id':'never-write','type':'function','function':{'name':'write_file','arguments':json.dumps({'path':'workspace/blocked-loop.txt','content':'must not execute'})}})
                    event({'choices':[{'delta':{'tool_calls':calls},'finish_reason':'tool_calls'}]})
            elif 'TRY_WRITE' in content:
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'write1','type':'function','function':{'name':'write_file','arguments':json.dumps({'path':'workspace/result.txt','content':'verified write'})}}]},'finish_reason':'tool_calls'}]})
            elif 'TRY_EDIT' in content:
                path='reference/context.txt' if 'READONLY' in content else 'workspace/result.txt'
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'edit1','type':'function','function':{'name':'edit_file','arguments':json.dumps({'path':path,'old_text':'verified write','new_text':'verified edit'})}}]},'finish_reason':'tool_calls'}]})
            elif 'READ_SECOND' in content:
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'read1','type':'function','function':{'name':'read_file','arguments':json.dumps({'path':'reference/context.txt'})}}]},'finish_reason':'tool_calls'}]})
            elif 'Ты — участник swarm' in content and '«slow»' in content:
                event({'choices':[{'delta':{'content':'REPORT from slow'}}]})
                for _ in range(20):time.sleep(.1);event({'choices':[{'delta':{'content':'.'}}]})
            elif 'Ты — участник swarm' in content and 'SWARM-FAIL' in content and '«risks»' in content and 'Это повтор' not in content:
                event({'error':{'message':'simulated rate limit'}})
            elif 'Ты — участник swarm' in content and 'SWARM-STEP' in content and '«scout»' in content and not any(m.get('role')=='tool' for m in messages):
                # First pass of a two-step member: one read-only tool call, so the
                # run reaches step 2 and the step has somewhere to live.
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'swread','type':'function','function':{'name':'read_file','arguments':json.dumps({'path':'reference/context.txt'})}}]},'finish_reason':'tool_calls'}]})
            elif 'Ты — участник swarm' in content:
                label=content.split('«')[1].split('»')[0] if '«' in content else 'member'
                event({'choices':[{'delta':{'content':'REPORT from '+label}}]})
                event({'choices':[{'delta':{'content':''},'finish_reason':'stop'}],'usage':{'prompt_tokens':5,'completion_tokens':6}})
            elif 'Ты — критик swarm-результата' in content:
                event({'choices':[{'delta':{'content':'MASTER merged answer (critic passed)'},'finish_reason':'stop'}],'usage':{'prompt_tokens':9,'completion_tokens':10}})
            elif 'Собери из них один итоговый ответ' in content:
                event({'choices':[{'delta':{'content':'MASTER merged answer'},'finish_reason':'stop'}],'usage':{'prompt_tokens':7,'completion_tokens':8}})
            else:event({'choices':[{'delta':{'content':'answer: '+content},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):pass

mock=ThreadingHTTPServer(('127.0.0.1',0),Mock);threading.Thread(target=mock.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory(prefix='allpaka-studio-test-') as tmp:
    root=pathlib.Path(tmp);workspace=root/'workspace';workspace.mkdir();reference=root/'reference';reference.mkdir();(reference/'context.txt').write_text('second-root-context')
    data=root/'history';data.mkdir()
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();base=f'http://127.0.0.1:{port}'
    env=dict(os.environ,ALLPAKA_ONLINE_WORKER='0',OPENROUTER_API_KEY='',ALLPAKA_LOCAL_BASE_URL=f'http://127.0.0.1:{mock.server_port}/v1')
    binary=pathlib.Path(os.environ.get('ALLPAKA_TEST_BINARY',str(pathlib.Path(__file__).resolve().parents[1]/'target/debug/allpaka')))
    log=open(root/'server.log','w')
    def start():
        p=subprocess.Popen([str(binary),'studio','--bind',f'127.0.0.1:{port}','--workspace',str(workspace),'--data-dir',str(data)],env=env,stdout=log,stderr=log)
        for _ in range(100):
            try:api('config');return p
            except Exception:
                if p.poll() is not None:raise RuntimeError((root/'server.log').read_text())
                time.sleep(.03)
        raise RuntimeError('Studio did not start')
    def api(path,body=None,headers=None):
        req=urllib.request.Request(base+'/api/'+path,data=None if body is None else json.dumps(body).encode(),headers=headers or {'Content-Type':'application/json','X-Allpaka-Client':'studio'})
        with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)
    def reject(path,body=None,status=400):
        try: api(path,body); raise AssertionError('request unexpectedly accepted')
        except urllib.error.HTTPError as error: assert error.code==status
    def wait(id,predicate,timeout=8):
        until=time.time()+timeout
        while time.time()<until:
            s=api('sessions/'+id)
            if predicate(s):return s
            time.sleep(.03)
        raise AssertionError(s)
    settings=dict(project_id='default',provider='local',model='mock',mode='chat',max_steps=8,allow_writes=False)
    def create(**kw):return api('sessions',dict(settings,**kw))['id']
    def trace_rows(session_id):return api('observability/traces?session_id='+session_id)['traces']
    def assert_trace_tree(trace):
        spans=trace['spans'];assert len({s['id'] for s in spans})==len(spans)
        roots=[s for s in spans if s['parent_id'] is None];assert len(roots)==1
        ids={s['id'] for s in spans}
        for span in spans:
            assert span['status']!='running' and span['duration_ms'] is not None,span
            assert span['parent_id'] is None or span['parent_id'] in ids and span['parent_id']<span['id']

    def act(id,kind,text='',**kw):return api(f'sessions/{id}/actions',dict(kind=kind,text=text,**kw))
    process=start()
    try:
        from allpaka_guardrails import policy_manifest, rules_from_manifest, check_guardrails, GuardrailBlocked
        policy_client=Studio(base)
        online_client=Studio(base);online_calls=len(requests)
        online_rule=dict(id='quality',project_id='default',evaluator_id='rubric',evaluator_version=1,sample_rate=0.5,enabled=True)
        online_snapshot=online_client.save_online_evaluation_rule(online_rule);online_hash=online_snapshot['rule_sha256']
        assert online_snapshot['kind']=='online_evaluation_rule' and online_snapshot['schema_version']==1 and online_snapshot['rule']==online_rule
        assert len(online_hash)==64 and online_client.save_online_evaluation_rule(online_rule)==online_snapshot
        assert online_client.online_evaluation_rule(online_hash)==online_snapshot
        online_next=online_client.save_online_evaluation_rule(dict(online_rule,evaluator_version=2));assert online_next['rule_sha256']!=online_hash
        assert online_client.online_evaluation_rule(online_hash)==online_snapshot
        for patch in [dict(project_id='missing'),dict(sample_rate=1.1),dict(evaluator_version=0),dict(enabled='yes'),dict(extra=True)]:
            reject('observability/online-rules',dict(online_rule,**patch))
        online_catalog=online_client.online_evaluation_rules('default',limit=1);online_catalog_next=online_client.online_evaluation_rules('default',offset=1,limit=1)
        assert online_catalog['kind']=='online_evaluation_rule_catalog' and online_catalog['total']==2 and online_catalog['has_more'] is True
        assert online_catalog_next['total']==2 and online_catalog_next['has_more'] is False
        assert {online_catalog['rules'][0]['rule_sha256'],online_catalog_next['rules'][0]['rule_sha256']}=={online_hash,online_next['rule_sha256']}
        assert online_catalog['rules'][0]['rule_sha256']<online_catalog_next['rules'][0]['rule_sha256']
        assert online_catalog['provider_calls']==0 and online_catalog['automatic_execution'] is False
        assert online_client.online_evaluation_rules('global')['total']==0 and len(requests)==online_calls
        reject('observability/online-rules?project_id=default&limit=101');reject('observability/online-rules?project_id=default&unknown=true')
        assert online_client.online_evaluation_binding('default','quality')['binding'] is None
        online_active=online_client.bind_online_evaluation_rule(online_hash,base_version=0,active=True)
        assert online_active['version']==1 and online_active['active'] is True and online_active['rule_sha256']==online_hash
        reject('observability/online-rule-bindings',dict(rule_sha256=online_hash,base_version=0,active=False))
        online_switched=online_client.bind_online_evaluation_rule(online_next['rule_sha256'],base_version=1,active=True)
        assert online_switched['version']==2 and online_switched['rule_sha256']==online_next['rule_sha256']
        online_inactive=online_client.bind_online_evaluation_rule(online_next['rule_sha256'],base_version=2,active=False)
        online_binding_status=online_client.online_evaluation_binding('default','quality')
        assert online_binding_status['binding']==online_inactive and online_inactive['version']==3 and online_inactive['active'] is False
        assert online_binding_status['provider_calls']==0 and online_binding_status['automatic_execution'] is False
        assert online_client.online_evaluation_binding('global','quality')['binding'] is None and len(requests)==online_calls
        reject('observability/online-rule-bindings',dict(rule_sha256=online_hash,base_version=3,active='yes'),status=422)
        online_binding_files=list((data/'observability'/'online-rules'/'bindings').glob('*/3.json'));assert len(online_binding_files)==1
        online_binding_path=online_binding_files[0];online_binding_original=online_binding_path.read_bytes()
        try:
            changed_binding=json.loads(online_binding_original);changed_binding['active']=True;online_binding_path.write_text(json.dumps(changed_binding))
            reject('observability/online-rule-bindings?project_id=default&rule_id=quality')
            reject('observability/online-rule-bindings',dict(rule_sha256=online_hash,base_version=3,active=True))
        finally:online_binding_path.write_bytes(online_binding_original)
        assert online_client.online_evaluation_binding('default','quality')==online_binding_status and len(requests)==online_calls
        online_path=data/'observability'/'online-rules'/(online_hash+'.json');online_original=online_path.read_bytes()
        try:
            altered_online=json.loads(online_original);altered_online['rule']['sample_rate']=1.0;online_path.write_text(json.dumps(altered_online))
            reject('observability/online-rules/'+online_hash,status=404);reject('observability/online-rules',online_rule)
            reject('observability/online-rules?project_id=default')
        finally:online_path.write_bytes(online_original)
        assert len(requests)==online_calls
        print('PASS HTTP immutable online rule snapshots, idempotent save, evaluator revisions, invalid admission and tamper rejection without inference',flush=True)
        policy_rules=[dict(id='literal',kind='forbidden_substrings',value=['private-marker']),dict(id='limit',kind='max_bytes',value=64000)]
        before_policy=len(requests)
        policy_receipt=policy_client.save_guardrail_policy(policy_rules)
        policy_hash=policy_receipt['policy']['policy_sha256']
        assert api('guardrail-policies/create',dict(rules=policy_rules))==policy_receipt
        for text in ['safe','private-marker']:
            for stage in ['input','output']:
                for action in ['block','observe']:
                    expected=check_guardrails(text,policy_rules,stage=stage,action='observe')
                    expected['action']=action;expected['blocked']=not expected['passed'] and action=='block'
                    assert policy_client.check_guardrail_policy(policy_hash,text,stage=stage,action=action)==expected
        for body in [dict(text='safe',stage='unknown',action='block'),dict(text='safe',stage='input',action='unknown'),dict(text='x'*64001,stage='input',action='observe')]:
            reject('guardrail-policies/'+policy_hash+'/check',body)
        for invalid in [dict(rules=[]),dict(rules=policy_rules,extra=True),dict(rules=[dict(id='limit',kind='max_bytes',value=True)])]:
            reject('guardrail-policies/create',invalid)
        assert policy_client.save_guardrail_policy(policy_rules)==policy_receipt
        assert policy_client.guardrail_policy(policy_hash)==policy_receipt
        assert check_guardrails('safe',rules_from_manifest(policy_receipt['policy']))['policy_sha256']==policy_hash
        altered=json.loads(json.dumps(policy_receipt['policy']));altered['rules'][1]['value']=1
        reject('guardrail-policies',altered)
        reject('guardrail-policies/'+('0'*64))
        reject('guardrail-policies/bad')
        for field,value in [('schema_version',True),('extra',1)]:
            invalid=dict(policy_receipt['policy']);invalid[field]=value;reject('guardrail-policies',invalid)
        duplicate=b'{"kind":"local_guardrail_policy","kind":"other"}'
        request=urllib.request.Request(base+'/api/guardrail-policies',data=duplicate,headers={'Content-Type':'application/json','X-Allpaka-Client':'studio'})
        try:urllib.request.urlopen(request,timeout=5);raise AssertionError('Duplicate keys accepted')
        except urllib.error.HTTPError as error:assert error.code==400
        policy_hashes={policy_hash}
        for limit in range(20):
            saved=policy_client.save_guardrail_policy([dict(id='limit',kind='max_bytes',value=limit)])
            policy_hashes.add(saved['policy']['policy_sha256'])
        first_page=policy_client.guardrail_policies(offset=0,limit=20)
        second_page=policy_client.guardrail_policies(offset=20,limit=20)
        assert first_page['total']==second_page['total']==21
        assert first_page['provider_calls']==second_page['provider_calls']==0
        rows=first_page['policies']+second_page['policies']
        assert len(first_page['policies'])==20 and len(second_page['policies'])==1
        assert [row['policy_sha256'] for row in rows]==sorted(policy_hashes)
        assert all(set(row)=={'policy_sha256','schema_version','rule_count'} for row in rows)
        assert policy_client.guardrail_policies(offset=21,limit=20)['policies']==[]
        for query in ('limit=0','limit=101','offset=10001','offset=-1','unknown=true'):
            reject('guardrail-policies?'+query)
        assert len(requests)==before_policy
        print('PASS native guardrail policy save/read, idempotence, SDK fingerprint, 20+1 catalog pages, malformed/tampered/duplicate rejection and no inference',flush=True)
        required_rules=[dict(id='required',kind='required_substrings',value=['Привет','ready'])]
        required_hash=policy_client.save_guardrail_policy(required_rules)['policy']['policy_sha256']
        for sample in ['Привет ready','Привет','привет ready']:
            assert policy_client.check_guardrail_policy(required_hash,sample,stage='input',action='observe')==check_guardrails(sample,required_rules)
        external_checks=[]
        for action in ['observe','block']:
            trace=policy_client.trace('default','rule-receipt-'+action,name='guarded_task')
            try:
                with trace:
                    check_guardrails('private-marker',policy_rules,action=action,trace=trace)
            except GuardrailBlocked:
                assert action=='block'
            assert trace.receipt and trace.export_error is None
            saved=api('observability/traces/'+trace.receipt['id'])
            child=saved['spans'][1];receipt=child['usage']['guardrail_receipt']
            assert receipt['passed'] is False and receipt['blocked']==(action=='block')
            assert receipt['rules']==[dict(rule_id='literal',kind='forbidden_substrings',passed=False),dict(rule_id='limit',kind='max_bytes',passed=True)]
            assert 'private-marker' not in json.dumps(saved)
            external_checks.append(trace.receipt['id'])
            body=dict(project_id='default',correlation_id='invalid-receipt',started_ms=trace.started_ms,spans=json.loads(json.dumps(trace.spans)))
            body['spans'][1]['usage']['guardrail_receipt']['text']='PRIVATE'
            reject('observability/external-traces',body)
            body['spans'][1]['usage']['guardrail_receipt'].pop('text');body['spans'][1]['name']='mismatched'
            reject('observability/external-traces',body)
        print('PASS external Python rule receipts exact persistence, blocked/observe status and private/mismatched rejection',flush=True)
        for action,blocked_stage in [('block','input'),('block','output'),('observe','output')]:
            input_hash=policy_hash
            output_hash=policy_client.save_guardrail_policy([dict(id='output_limit',kind='max_bytes',value=0)])['policy']['policy_sha256']
            guarded=create(guardrails=dict(input_policy_sha256=input_hash,output_policy_sha256=output_hash,action=action))
            before_guard=len(requests);act(guarded,'send','private-marker' if blocked_stage=='input' else 'guard fixture')
            state=wait(guarded,lambda state:state['status'] in ('idle','error'))
            if action=='block':
                assert state['status']=='error' and state['error']=='guardrail_blocked'
                assert not any(message['content'] for message in state['messages'] if message['role']=='assistant')
                assert len(requests)-before_guard==(0 if blocked_stage=='input' else 1)
                if blocked_stage=='output':assert state['usage']['prompt_tokens']==3 and state['usage']['completion_tokens']==4
            else:
                assert state['status']=='idle' and any(message['content'] for message in state['messages'] if message['role']=='assistant')
            guarded_trace=trace_rows(guarded)[0]
            failed_check=next(span for span in guarded_trace['spans'] if span['name'].startswith('guardrail.'+blocked_stage+'.'+action+'.fail.'))
            check_receipt=failed_check['usage']['guardrail_receipt']
            assert check_receipt['passed'] is False and check_receipt['blocked']==(action=='block')
            assert all(set(rule)=={'rule_id','kind','passed'} for rule in check_receipt['rules'])
            assert 'private-marker' not in json.dumps(check_receipt)
        guarded_roster=dict(members=[dict(label='a',role='',provider='local',model='mock'),dict(label='b',role='',provider='local',model='mock')],max_steps_per_member=1)
        for blocked in [True,False]:
            out_hash=output_hash if blocked else policy_hash
            guarded_swarm=create(mode='swarm',swarm=dict(guarded_roster,critic=not blocked),guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=out_hash,action='block'))
            act(guarded_swarm,'send','guard swarm fixture');swarm_state=wait(guarded_swarm,lambda state:state['status'] in ['idle','error'])
            reports=next(message['swarm'] for message in swarm_state['messages'] if message.get('swarm'))
            if blocked:
                assert swarm_state['status']=='error' and all(not report['content'] for report in reports)
            else:
                assert swarm_state['status']=='idle' and all(report['content'] for report in reports)
                assert any(message['content'] for message in swarm_state['messages'] if message['role']=='assistant')
        accepted_swarm_trace=trace_rows(guarded_swarm)[0]
        assert any(span['kind']=='critic' for span in accepted_swarm_trace['spans'])
        checks=[span for span in accepted_swarm_trace['spans'] if span['name'].startswith('guardrail.')]
        assert len(checks)>=9 and all(span['status']=='completed' for span in checks)
        before_retry=len(requests);before_traces=len(trace_rows(guarded_swarm));act(guarded_swarm,'retry_member','a')
        retried_guard=wait(guarded_swarm,lambda state:state['status']=='idle' and len(requests)>=before_retry+3)
        assert len(trace_rows(guarded_swarm))==before_traces+1
        retry_checks=[span for span in trace_rows(guarded_swarm)[0]['spans'] if span['name'].startswith('guardrail.')]
        assert len(retry_checks)>=7 and all(span['status']=='completed' for span in retry_checks)
        assert any(message['content'] for message in retried_guard['messages'] if message['role']=='assistant')
        print('PASS native local Swarm blocked reports stay empty and accepted reports/synthesis publish',flush=True)
        streaming_guard=create(guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=output_hash,action='block'))
        guard_stream_started.clear();act(streaming_guard,'send','GUARD_STREAM_TEST')
        assert guard_stream_started.wait(5)
        for _ in range(5):
            in_flight=api('sessions/'+streaming_guard)
            assert not any(message['content'] or message.get('reasoning_content') for message in in_flight['messages'] if message['role']=='assistant')
            time.sleep(.05)
        final_guard=wait(streaming_guard,lambda state:state['status']=='error')
        assert final_guard['error']=='guardrail_blocked'
        assert 'BLOCKED_STREAM_TEXT' not in json.dumps(final_guard)
        cancelled_guard=create(guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=output_hash,action='block'))
        guard_stream_started.clear();act(cancelled_guard,'send','GUARD_STREAM_TEST');assert guard_stream_started.wait(5)
        act(cancelled_guard,'stop');cancelled_state=wait(cancelled_guard,lambda state:state['status']=='paused')
        assert 'BLOCKED_STREAM_TEXT' not in json.dumps(cancelled_state)
        cancelled_trace=trace_rows(cancelled_guard)[0]
        assert cancelled_trace['status']=='interrupted' and all(span['status']!='running' for span in cancelled_trace['spans'])
        for mode in ['plan','auto','goal']:
            for stage in ['input','output']:
                guarded_mode=create(mode=mode,guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=output_hash,action='block'))
                before_mode=len(requests);act(guarded_mode,'send','private-marker' if stage=='input' else 'guard mode fixture')
                mode_state=wait(guarded_mode,lambda state:state['status']=='error')
                assert mode_state['error']=='guardrail_blocked'
                assert len(requests)-before_mode==(0 if stage=='input' else 1)
                assert not any(message['content'] for message in mode_state['messages'] if message['role']=='assistant')
        reject('sessions',dict(settings,mode='swarm',guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=output_hash,action='block')))
        print('PASS native Chat guardrails input no inference, buffered blocked output and observe delivery',flush=True)
        duplicate=subprocess.run([str(binary),'studio','--bind','127.0.0.1:0','--workspace',str(workspace),'--data-dir',str(data)],env=env,capture_output=True,text=True,timeout=5)
        assert duplicate.returncode!=0 and 'Another Studio process owns this data directory' in duplicate.stderr
        print('PASS exclusive data-directory writer lock rejects a second Studio process')
        assert len(api('config')['providers'])==8
        catalog=api('providers/local/models')['catalog'][0]
        assert catalog['context_length']==32000 and catalog['pricing']['prompt']=='0.000001'
        assert catalog['architecture']['input_modalities']==['text','image']
        assert api('providers/local/models')['models']==['mock']
        doctor_before=len(requests)
        doctor=Studio(base).provider_doctor('local','mock')
        assert doctor['status']=='ready' and doctor['model_count']==1 and doctor['generation_calls']==0 and doctor['inference_verified'] is False
        assert Studio(base).provider_doctor('local','unknown')['status']=='model_missing'
        for suffix,status in [('auth','unauthorized'),('limit','rate_limited'),('redirect','redirect_rejected'),('missing','catalog_unavailable'),('invalid','invalid_catalog'),('large','catalog_too_large'),('empty','empty_catalog')]:
            provider=api('providers',dict(name='Doctor fixture',base=f'http://127.0.0.1:{mock.server_address[1]}/doctor-'+suffix))
            diagnostic=Studio(base).provider_doctor(provider['id'])
            assert diagnostic['status']==status and 'PROVIDER_SECRET_BODY' not in json.dumps(diagnostic)
            api('providers/'+provider['id']+'/delete',{})
        reject('providers/local/doctor',dict(model='x'*201))
        assert len(requests)==doctor_before
        print('PASS provider doctor catalog readiness, missing model, HTTP diagnoses, bounded/strict JSON, rejected redirects and no inference/body disclosure')
        external=dict(project_id='default',correlation_id='python-fixture',started_ms=1000,spans=[
            dict(parent_id=None,kind='agent',name='pipeline',status='completed',started_ms=1000,duration_ms=100),
            dict(parent_id=0,kind='model',name='mock',status='completed',started_ms=1010,duration_ms=50,
                 usage=dict(input_tokens=7,output_tokens=9,cost=0.25,cost_currency='USD'))])
        before=len(requests)
        external_receipt=Studio(base).ingest_trace('default','python-fixture',1000,external['spans'])
        adapter_client=Studio(base);adapter_calls=[]
        adapter_response={'usage':{'prompt_tokens':12,'completion_tokens':4,'prompt_tokens_details':{'cached_tokens':8}},'choices':'PRIVATE_COMPLETION'}
        def adapter_create(**kwargs):adapter_calls.append(kwargs);return adapter_response
        adapter=adapter_client.track_openai_chat(adapter_create,provider_id='local',model_id='adapter-model')
        with adapter_client.trace('default','openai-adapter') as adapter_trace:
            assert adapter(model='adapter-model',messages=['PRIVATE_PROMPT']) is adapter_response
        adapter_record=api('observability/traces/'+adapter_trace.receipt['id'])
        assert adapter_record['spans'][1]['kind']=='external_model'
        assert adapter_record['spans'][1]['usage']['input_tokens']==12 and adapter_record['spans'][1]['usage']['cache_read_input_tokens']==8
        assert 'PRIVATE_' not in json.dumps(adapter_record)
        adapter_summary=adapter_client.trace_summary(session_id='external-openai-adapter')
        assert adapter_summary['models'][0]['model']=='adapter-model' and adapter_summary['models'][0]['provider_id']=='local'
        assert adapter_summary['models'][0]['output_tokens']==4
        assert len(adapter_calls)==1 and len(requests)==before
        response_adapter_result={'usage':{'input_tokens':9,'output_tokens':2,'input_tokens_details':{'cached_tokens':4}},'output':'PRIVATE_RESPONSES_OUTPUT'}
        response_adapter=adapter_client.track_openai_responses(lambda **kwargs:response_adapter_result,provider_id='local',model_id='responses-model')
        with adapter_client.trace('default','responses-adapter') as responses_trace:
            assert response_adapter(model='responses-model',input='PRIVATE_RESPONSES_INPUT') is response_adapter_result
        class AdapterStream:
            def __init__(self):self.chunks=iter([{'choices':'PRIVATE_STREAM_TEXT'},{'usage':{'prompt_tokens':11,'completion_tokens':5}}]);self.closed=False
            def __iter__(self):return self
            def __next__(self):return next(self.chunks)
            def close(self):self.closed=True
        raw_adapter_stream=AdapterStream()
        stream_adapter=adapter_client.track_openai_chat_stream(lambda **kwargs:raw_adapter_stream,provider_id='local',model_id='stream-model')
        with adapter_client.trace('default','stream-adapter') as stream_trace:
            with stream_adapter(model='stream-model',stream=True) as stream:
                assert len(list(stream))==2
        assert raw_adapter_stream.closed
        stream_record=api('observability/traces/'+stream_trace.receipt['id'])
        assert stream_record['spans'][1]['status']=='completed' and stream_record['spans'][1]['usage']['output_tokens']==5
        assert 'PRIVATE_' not in json.dumps(stream_record)
        assert len(requests)==before
        import asyncio
        class AsyncAdapterStream:
            def __init__(self):self.chunks=iter([{'choices':'PRIVATE_ASYNC_STREAM'},{'usage':{'prompt_tokens':6,'completion_tokens':3}}]);self.closed=False
            def __aiter__(self):return self
            async def __anext__(self):
                try:return next(self.chunks)
                except StopIteration:raise StopAsyncIteration
            async def close(self):self.closed=True
        async def verify_async_stream():
            raw=AsyncAdapterStream()
            async def create(**kwargs):return raw
            wrapped=adapter_client.track_openai_chat_stream(create,provider_id='local',model_id='async-stream-model')
            with adapter_client.trace('default','async-stream-adapter') as trace:
                async with await wrapped(model='async-stream-model',stream=True) as stream:
                    assert len([chunk async for chunk in stream])==2
            assert raw.closed
            return trace.receipt['id']
        async_stream_record=api('observability/traces/'+asyncio.run(verify_async_stream()))
        assert async_stream_record['spans'][1]['status']=='completed'
        assert async_stream_record['spans'][1]['usage']['input_tokens']==6 and async_stream_record['spans'][1]['usage']['output_tokens']==3
        assert 'PRIVATE_' not in json.dumps(async_stream_record) and len(requests)==before
        async def verify_responses_event_stream(status):
            raw=AsyncAdapterStream()
            raw.chunks=iter([{'type':'response.output_text.delta','delta':'PRIVATE_EVENT_DELTA'},
                {'type':'response.'+status,'response':{'status':status,'usage':{'input_tokens':8,'output_tokens':4},'output':'PRIVATE_EVENT_OUTPUT'}}])
            async def create(**kwargs):return raw
            wrapped=adapter_client.track_openai_responses_stream(create,provider_id='local',model_id='responses-stream-model')
            with adapter_client.trace('default','responses-stream-'+status) as trace:
                async with await wrapped(model='responses-stream-model',stream=True) as stream:
                    assert len([event async for event in stream])==2
            assert raw.closed
            return trace.receipt['id']
        for event_status in ('completed','failed','incomplete'):
            event_record=api('observability/traces/'+asyncio.run(verify_responses_event_stream(event_status)))
            assert event_record['spans'][1]['status']==('completed' if event_status=='completed' else 'failed')
            assert event_record['spans'][1]['usage']['input_tokens']==8 and event_record['spans'][1]['usage']['output_tokens']==4
            assert 'PRIVATE_' not in json.dumps(event_record)
        assert len(requests)==before
        anthropic_result={'usage':{'input_tokens':12,'output_tokens':3,'cache_creation_input_tokens':4,'cache_read_input_tokens':9},'content':'PRIVATE_ANTHROPIC_CONTENT'}
        anthropic_adapter=adapter_client.track_anthropic_messages(lambda **kwargs:anthropic_result,provider_id='anthropic',model_id='claude-fixture')
        with adapter_client.trace('default','anthropic-adapter') as anthropic_trace:
            assert anthropic_adapter(model='claude-fixture',messages='PRIVATE_ANTHROPIC_INPUT') is anthropic_result
        async def verify_anthropic_event_stream(complete):
            raw=AsyncAdapterStream()
            events=[{'type':'message_start','message':{'type':'message','usage':{'input_tokens':12,'output_tokens':1,'cache_creation_input_tokens':4,'cache_read_input_tokens':9},'content':'PRIVATE_ANTHROPIC_STREAM'}},
                {'type':'message_delta','usage':{'output_tokens':4}},
                {'type':'message_delta','usage':{'output_tokens':7}}]
            if complete:events.append({'type':'message_stop'})
            raw.chunks=iter(events)
            async def create(**kwargs):return raw
            wrapped=adapter_client.track_anthropic_messages_stream(create,provider_id='anthropic',model_id='claude-stream-fixture')
            with adapter_client.trace('default','anthropic-events-'+str(complete)) as trace:
                async with await wrapped(model='claude-stream-fixture',stream=True) as stream:
                    assert len([event async for event in stream])==len(events)
            assert raw.closed
            return trace.receipt['id']
        for complete in (True,False):
            anthropic_stream_record=api('observability/traces/'+asyncio.run(verify_anthropic_event_stream(complete)))
            assert anthropic_stream_record['spans'][1]['status']==('completed' if complete else 'interrupted')
            assert anthropic_stream_record['spans'][1]['usage']['input_tokens']==25 and anthropic_stream_record['spans'][1]['usage']['output_tokens']==7
            assert 'PRIVATE_' not in json.dumps(anthropic_stream_record) and len(requests)==before
        anthropic_record=api('observability/traces/'+anthropic_trace.receipt['id'])
        assert anthropic_record['spans'][1]['usage']['input_tokens']==25
        assert anthropic_record['spans'][1]['usage']['output_tokens']==3 and anthropic_record['spans'][1]['usage']['cache_read_input_tokens']==9
        assert 'PRIVATE_' not in json.dumps(anthropic_record)
        anthropic_summary=adapter_client.trace_summary(session_id='external-anthropic-adapter')
        assert anthropic_summary['models'][0]['provider_id']=='anthropic' and anthropic_summary['models'][0]['model']=='claude-fixture'
        assert anthropic_summary['models'][0]['input_tokens']==25 and len(requests)==before
        assert anthropic_record['spans'][1]['usage']['cache_creation_input_tokens']==4
        assert anthropic_summary['cache_creation_input_tokens']==4 and anthropic_summary['cache_read_input_tokens']==9
        assert anthropic_summary['models'][0]['cache_creation_input_tokens']==4
        assert anthropic_summary['cache_creation_unknown_calls']==0 and anthropic_summary['cache_read_unknown_calls']==0
        responses_record=api('observability/traces/'+responses_trace.receipt['id'])
        assert responses_record['spans'][1]['usage']['input_tokens']==9 and responses_record['spans'][1]['usage']['output_tokens']==2
        assert responses_record['spans'][1]['usage']['cache_read_input_tokens']==4
        assert 'PRIVATE_' not in json.dumps(responses_record)
        responses_summary=adapter_client.trace_summary(session_id='external-responses-adapter')
        assert responses_summary['models'][0]['model']=='responses-model' and responses_summary['models'][0]['provider_id']=='local'
        assert len(requests)==before
        external_record=api('observability/traces/'+external_receipt['id'])
        assert external_record['spans'][1]['kind']=='external_model' and external_record['spans'][1]['parent_id']==0
        assert external_record['spans'][1]['duration_ms']==50 and external_record['spans'][1]['usage']['output_tokens']==9
        assert external_record['session_id']=='external-python-fixture' and external_receipt['provider_calls']==0
        ext_summary=Studio(base).trace_summary(session_id='external-python-fixture')
        assert ext_summary['external_model_calls']==1 and ext_summary['reported_cost_by_currency']['USD']==0.25
        assert ext_summary['models'][0]['source']=='external' and ext_summary['models'][0]['model']=='mock'
        assert ext_summary['models'][0]['calls']==1 and ext_summary['models'][0]['duration_p95_ms']==50
        assert ext_summary['models'][0]['input_tokens']==7
        assert ext_summary['cache_creation_unknown_calls']==1 and ext_summary['cache_read_unknown_calls']==1
        keyed=Studio(base).ingest_trace('default','retry-fixture',1000,external['spans'],idempotency_key='retry-key')
        retry=Studio(base).ingest_trace('default','retry-fixture',1000,external['spans'],idempotency_key='retry-key')
        assert keyed['id']==retry['id'] and not keyed['deduplicated'] and retry['deduplicated']
        selection_calls=len(requests)
        online_client.bind_online_evaluation_rule(online_next['rule_sha256'],base_version=3,active=True)
        online_selection=online_client.select_online_evaluations('default',external_receipt['id'])
        assert online_selection['kind']=='online_evaluation_selection' and online_selection['trace_id']==external_receipt['id']
        assert len(online_selection['decisions'])==1 and type(online_selection['decisions'][0]['selected']) is bool
        assert online_selection['schema_version']==2 and len(online_selection['trace_sha256'])==64
        pinned_trace_path=data/'observability'/'traces'/(external_receipt['id']+'.json');pinned_trace_original=pinned_trace_path.read_bytes()
        try:
            changed_trace=json.loads(pinned_trace_original);changed_trace['spans'][1]['name']='changed-metadata';pinned_trace_path.write_text(json.dumps(changed_trace))
            reject('observability/online-selections',dict(project_id='default',trace_id=external_receipt['id']))
        finally:pinned_trace_path.write_bytes(pinned_trace_original)
        assert online_client.select_online_evaluations('default',external_receipt['id'])==online_selection and len(requests)==selection_calls

        assert online_selection['decisions'][0]['binding']['version']==4 and online_selection['decisions'][0]['binding']['rule_sha256']==online_next['rule_sha256']
        assert online_selection['provider_calls']==0 and online_selection['automatic_execution'] is False
        online_client.bind_online_evaluation_rule(online_next['rule_sha256'],base_version=4,active=False)
        online_binding_status=online_client.online_evaluation_binding('default','quality')
        assert online_client.select_online_evaluations('default',external_receipt['id'])==online_selection
        assert online_client.select_online_evaluations('default',keyed['id'])['decisions']==[]
        reject('observability/online-selections',dict(project_id='global',trace_id=external_receipt['id']))
        reject('observability/online-selections',dict(project_id='default',trace_id='missing-trace'))
        selection_directory=data/'observability'/'online-rules'/'selections'
        selection_path=next(path for path in selection_directory.glob('*.json') if json.loads(path.read_text())['trace_id']==external_receipt['id'])
        selection_original=selection_path.read_bytes()
        online_archive=online_client.online_evaluation_selection_archive('default',external_receipt['id'])
        assert online_archive['selection']==online_selection and online_archive['trace_evidence_pinned'] is True and online_archive['execution_eligible'] is False and online_archive['provider_calls']==0
        assert selection_path.read_bytes()==selection_original
        try:
            legacy_selection=dict(online_selection);legacy_selection.pop('trace_sha256');legacy_selection['schema_version']=1
            import hashlib
            legacy_selection['selection_sha256']=hashlib.sha256(json.dumps(['default',external_receipt['id'],legacy_selection['decisions']],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            legacy_bytes=json.dumps(legacy_selection).encode();selection_path.write_bytes(legacy_bytes)
            legacy_archive=online_client.online_evaluation_selection_archive('default',external_receipt['id'])
            assert legacy_archive['selection']==legacy_selection and legacy_archive['trace_evidence_pinned'] is False and legacy_archive['execution_eligible'] is False
            assert selection_path.read_bytes()==legacy_bytes and len(requests)==selection_calls
            reject('observability/online-selections',dict(project_id='default',trace_id=external_receipt['id']))
        finally:selection_path.write_bytes(selection_original)
        assert online_client.online_evaluation_selection_archive('default',external_receipt['id'])==online_archive

        try:
            altered_selection=json.loads(selection_original);altered_selection['decisions'][0]['selected']=not altered_selection['decisions'][0]['selected'];selection_path.write_text(json.dumps(altered_selection))
            reject('observability/online-selections',dict(project_id='default',trace_id=external_receipt['id']))
            selection_path.write_bytes(b' '* (1024*1024+1))
            reject('observability/online-selections',dict(project_id='default',trace_id=external_receipt['id']))
        finally:selection_path.write_bytes(selection_original)
        selection_existing=set(selection_directory.glob('*.json'));selection_capacity_files=[]
        try:
            for index in range(1000-len(selection_existing)):
                path=selection_directory/('quota-'+str(index)+'.json');path.write_text('{}');selection_capacity_files.append(path)
            reject('observability/online-selections',dict(project_id='default',trace_id=adapter_trace.receipt['id']))
            assert online_client.select_online_evaluations('default',external_receipt['id'])==online_selection
        finally:
            for path in selection_capacity_files:path.unlink()
        assert set(selection_directory.glob('*.json'))==selection_existing and len(requests)==selection_calls
        assert len(requests)==selection_calls
        print('PASS HTTP online selections pin active binding, retain decisions after deactivation, apply current state to new traces and reject foreign/missing targets without inference',flush=True)
        unknown_snapshot=online_client.save_online_evaluation_rule(dict(online_rule,sample_rate=1.0))
        online_client.bind_online_evaluation_rule(unknown_snapshot['rule_sha256'],base_version=5,active=True)
        unknown_assessment=online_client.assess_online_trace('default',responses_trace.receipt['id'])
        assert unknown_assessment['assessments'][0]['status']=='failed' and unknown_assessment['assessments'][0]['error_code']=='evaluator_unavailable' and unknown_assessment['assessments'][0]['scores']=={}
        assert online_client.assess_online_trace('default',responses_trace.receipt['id'])==unknown_assessment
        health_snapshot=online_client.save_online_evaluation_rule(dict(online_rule,evaluator_id='trace_health',sample_rate=1.0))
        online_client.bind_online_evaluation_rule(health_snapshot['rule_sha256'],base_version=6,active=True)
        health_assessment=online_client.assess_online_trace('default',adapter_trace.receipt['id'])
        assert health_assessment['assessment_source']=='native_trace_metadata' and health_assessment['provider_calls']==0 and health_assessment['automatic_execution'] is False
        assert health_assessment['assessments'][0]['status']=='completed' and health_assessment['assessments'][0]['scores']=={'span_success_rate':1.0}
        assert health_assessment['assessments'][0]['evaluator_id']=='trace_health' and health_assessment['assessments'][0]['evaluator_version']==1
        assert online_client.assess_online_trace('default',adapter_trace.receipt['id'])==health_assessment and len(requests)==selection_calls
        auto_calls=len(requests);auto_session=create();act(auto_session,'send','ONLINE_AUTO_QUEUE_FIXTURE')
        wait(auto_session,lambda state:state['status']=='idle' and len(state['messages'])==2)
        online_auto_trace=trace_rows(auto_session)[0];assert online_auto_trace['status']=='completed'
        auto_jobs=data/'observability'/'online-rules'/'jobs'
        until=time.time()+5;auto_job=None
        while time.time()<until:
            if auto_jobs.exists():
                for path in auto_jobs.glob('*.json'):
                    candidate=json.loads(path.read_text())
                    if candidate.get('trace_id')==online_auto_trace['id']:auto_job=candidate;break
            if auto_job:break
            time.sleep(.02)
        assert auto_job and auto_job['status']=='pending' and auto_job['provider_calls']==0 and len(requests)==auto_calls+1
        auto_selection=online_client.online_evaluation_selection_archive('default',online_auto_trace['id'])['selection']
        assert auto_selection['selection_sha256']==auto_job['selection_sha256'] and auto_selection['trace_sha256']==auto_job['trace_sha256']
        assert auto_selection['decisions'][0]['binding']['version']==7 and auto_selection['decisions'][0]['selected'] is True
        auto_existing=set(auto_jobs.glob('*.json'));auto_quota_files=[]
        try:
            for index in range(1000-len(auto_existing)):
                path=auto_jobs/('quota-'+str(index)+'.json');path.write_text('{}');auto_quota_files.append(path)
            quota_failures=(root/'server.log').read_text().count('Online job capacity reached')
            full_queue_session=create();act(full_queue_session,'send','ONLINE_FULL_QUEUE_FIXTURE')
            wait(full_queue_session,lambda state:state['status']=='idle' and len(state['messages'])==2)
            full_queue_trace=trace_rows(full_queue_session)[0];assert full_queue_trace['status']=='completed'
            until=time.time()+5
            while time.time()<until and (root/'server.log').read_text().count('Online job capacity reached')==quota_failures:time.sleep(.02)
            assert (root/'server.log').read_text().count('Online job capacity reached')>quota_failures
            assert len(list(auto_jobs.glob('*.json')))==1000 and len(requests)==auto_calls+2
        finally:
            for path in auto_quota_files:path.unlink()
        assert set(auto_jobs.glob('*.json'))==auto_existing
        processed=online_client.drain_online_evaluation_jobs(limit=1)
        assert processed['completed']==1 and processed['failed']==0 and processed['provider_calls']==0
        processed_again=online_client.drain_online_evaluation_jobs(limit=20)
        assert processed_again['completed']==0 and processed_again['failed']==0 and processed_again['already_finished']==1
        online_evidence_session=create();act(online_evidence_session,'send','ONLINE_CHANGED_JOB_FIXTURE')
        wait(online_evidence_session,lambda state:state['status']=='idle' and len(state['messages'])==2)
        online_evidence_trace=trace_rows(online_evidence_session)[0];until=time.time()+5
        while time.time()<until and len(list(auto_jobs.glob('*.json')))<2:time.sleep(.02)
        assert len(list(auto_jobs.glob('*.json')))==2
        online_evidence_path=data/'observability'/'traces'/(online_evidence_trace['id']+'.json');online_evidence_original=online_evidence_path.read_bytes()
        try:
            changed_evidence=json.loads(online_evidence_original);changed_evidence['spans'][0]['name']='changed-after-admission';online_evidence_path.write_text(json.dumps(changed_evidence))
            failed_batch=online_client.drain_online_evaluation_jobs(limit=20);assert failed_batch['failed']==1 and failed_batch['completed']==0 and failed_batch['already_finished']==1
        finally:online_evidence_path.write_bytes(online_evidence_original)
        terminal_batch=online_client.drain_online_evaluation_jobs(limit=20)
        assert terminal_batch['already_finished']==2 and terminal_batch['completed']==terminal_batch['failed']==0 and len(requests)==auto_calls+3
        online_result_files={path.name:path.read_bytes() for path in (data/'observability'/'online-rules'/'job-results').glob('*.json')}
        assert len(online_result_files)==2
        reject('observability/online-jobs/drain',dict(limit=101))
        print('PASS HTTP online job execution, finished-job skip, changed-evidence terminal failure and no evaluator replay',flush=True)

        before=len(requests)  # Explicit chat fixtures above account for their three model requests.
        print('PASS HTTP native chat completion admits pinned pending online job; full queue leaves primary answer completed',flush=True)
        online_client.bind_online_evaluation_rule(health_snapshot['rule_sha256'],base_version=7,active=False)
        online_binding_status=online_client.online_evaluation_binding('default','quality')
        online_catalog=online_client.online_evaluation_rules('default',limit=1);online_catalog_next=online_client.online_evaluation_rules('default',offset=1,limit=1)
        print('PASS HTTP explicit native health assessment, unavailable evaluator failure and pinned idempotent results without inference',flush=True)


        reject('observability/external-traces',dict(external,correlation_id='changed',idempotency_key='retry-key'),status=409)
        assert Studio(base).trace_summary(session_id='external-retry-fixture')['trace_count']==1
        keyed_record=api('observability/traces/'+keyed['id'])
        export_feedback=api('observability/traces/'+keyed['id']+'/feedback',dict(base_version=0,annotation=dict(author='export-reviewer',metric='accuracy',value=0.75,comment='EXPLICIT_EXPORT_COMMENT')))
        metadata_export=Studio(base).export_trace(keyed['id'])
        assert metadata_export['kind']=='trace_export' and metadata_export['schema_version']==1
        assert metadata_export['trace']==keyed_record and metadata_export['feedback'] is None
        assert 'EXPLICIT_EXPORT_COMMENT' not in json.dumps(metadata_export) and metadata_export['provider_calls']==0
        reviewed_export=Studio(base).export_trace(keyed['id'],include_feedback=True)
        bulk=Studio(base).export_traces('default',[keyed['id'],external_receipt['id']])
        assert bulk['kind']=='trace_export_batch' and bulk['schema_version']==1 and bulk['trace_count']==2
        assert [item['trace']['id'] for item in bulk['traces']]==[keyed['id'],external_receipt['id']]
        assert all(item['feedback'] is None for item in bulk['traces']) and bulk['provider_calls']==0
        bulk_feedback=Studio(base).export_traces('default',[keyed['id']],include_feedback=True)
        assert bulk_feedback['traces'][0]['trace']==reviewed_export['trace']
        assert bulk_feedback['traces'][0]['feedback']==reviewed_export['feedback']
        trace_export_path=root/'selected-traces.json'
        trace_export_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'export-studio-traces.py'),'--base-url',base,'--trace-id',keyed['id'],'--trace-id',external_receipt['id'],'--include-feedback','--output',str(trace_export_path)]
        trace_export_done=subprocess.run(trace_export_cli,capture_output=True,text=True,timeout=15)
        assert trace_export_done.returncode==0,trace_export_done.stderr+trace_export_done.stdout
        saved_batch=json.loads(trace_export_path.read_text())
        assert [item['trace']['id'] for item in saved_batch['traces']]==[keyed['id'],external_receipt['id']]
        assert saved_batch['traces'][0]['feedback']==reviewed_export['feedback']
        preserved_export=trace_export_path.read_bytes()
        duplicate_export=subprocess.run(trace_export_cli,capture_output=True,text=True,timeout=15)
        assert duplicate_export.returncode==2 and trace_export_path.read_bytes()==preserved_export

        reject('observability/trace-exports',dict(project_id='other',trace_ids=[keyed['id']]))
        reject('observability/trace-exports',dict(project_id='default',trace_ids=[keyed['id'],keyed['id']]))
        reject('observability/trace-exports',dict(project_id='default',trace_ids=[]))
        reject('observability/trace-exports',dict(project_id='default',trace_ids=['missing-trace']))

        assert reviewed_export['feedback']==export_feedback and reviewed_export['privacy']['feedback_included']
        assert Studio(base).remove_trace(keyed['id'])['removed']
        reject('observability/online-selections',dict(project_id='default',trace_id=keyed['id']))
        reject('observability/trace-exports',dict(project_id='default',trace_ids=[external_receipt['id'],keyed['id']]))
        reject('observability/traces/'+keyed['id'])
        reject('observability/traces/'+keyed['id']+'/export')
        assert Studio(base).trace_summary(session_id='external-retry-fixture')['trace_count']==0
        assert any(row['id']==keyed['id'] for row in api('observability/traces?removed=true')['traces'])
        reject('observability/external-traces',dict(external,correlation_id='retry-fixture',idempotency_key='retry-key'))
        assert not Studio(base).restore_trace(keyed['id'])['removed']
        assert api('observability/traces/'+keyed['id'])==keyed_record
        assert Studio(base).export_trace(keyed['id'],include_feedback=True)['feedback']==export_feedback
        print('PASS versioned trace export preserves exact metadata, includes human feedback only explicitly and rejects removed traces')
        assert Studio(base).trace_summary(session_id='external-retry-fixture')['trace_count']==1
        print('PASS recoverable trace removal, hidden direct reads/totals, removed catalog and explicit restoration without ingestion resurrection')
        reject('observability/external-traces',dict(external,project_id='foreign'))
        reject('observability/external-traces',dict(external,spans=[dict(external['spans'][0],prompt='PRIVATE_PROMPT')]),status=422)
        reject('observability/external-traces',dict(external,spans=[external['spans'][0],dict(external['spans'][1],parent_id=1)]))
        reject('observability/external-traces',dict(external,spans=[dict(external['spans'][0],status='running')]))
        reject('observability/external-traces',dict(external,spans=[external['spans'][0],dict(external['spans'][1],usage=dict(api_key='PRIVATE_KEY'))]),status=422)
        assert len(requests)==before
        print('PASS external SDK trace ingestion, bounded tree metadata, reported cost summary and rejected private fields without model calls')
        with Studio(base).trace('default','context-fixture') as external_context:
            with external_context.span('retrieval','tool'):
                with external_context.span('model-v1','model') as model_span:
                    model_span.set_usage(input_tokens=5,output_tokens=6)
        context_record=api('observability/traces/'+external_context.receipt['id'])
        assert [span['parent_id'] for span in context_record['spans']]==[None,0,1]
        assert all(span['status']=='completed' and span['duration_ms'] is not None for span in context_record['spans'])
        original_error=RuntimeError('PRIVATE_EXCEPTION_BODY')
        try:
            with Studio(base).trace('default','context-error') as errored_context:
                with errored_context.span('tool-v1'):raise original_error
        except RuntimeError as error:assert error is original_error
        error_record=api('observability/traces/'+errored_context.receipt['id'])
        assert error_record['status']=='failed' and error_record['spans'][1]['status']=='failed'
        assert 'PRIVATE_EXCEPTION_BODY' not in json.dumps(error_record) and len(requests)==before
        print('PASS SDK trace contexts persist nested timing/status, retain business exceptions and omit exception text')
        tracked_client=Studio(base)
        @tracked_client.track('lookup','tool')
        def tracked_lookup(value):return value
        @tracked_client.track('pipeline-step','agent')
        def tracked_step(value):return tracked_lookup(value)
        with tracked_client.trace('default','decorator-fixture') as decorated_trace:
            assert tracked_step('PRIVATE_TRACKED_VALUE')=='PRIVATE_TRACKED_VALUE'
        decorated_record=api('observability/traces/'+decorated_trace.receipt['id'])
        assert [span['parent_id'] for span in decorated_record['spans']]==[None,0,1]
        assert 'PRIVATE_TRACKED_VALUE' not in json.dumps(decorated_record) and len(requests)==before
        print('PASS function tracking decorators persist native external hierarchy without function payload capture')
        online_calls=len(requests)
        @tracked_client.track('online-task',evaluate=lambda result:{'exact':float(result=='PRIVATE_ONLINE_RESULT')},evaluator_id='exact-check',evaluator_version=2)
        def online_task():return 'PRIVATE_ONLINE_RESULT'
        @tracked_client.track('online-bad',evaluate=lambda result:{'invalid':float('nan')},evaluator_id='invalid-check',evaluator_version=3)
        def online_bad():return 'PRIVATE_ONLINE_RESULT'
        with tracked_client.trace('default','online-evaluation-fixture') as online_trace:
            assert online_task()=='PRIVATE_ONLINE_RESULT'
            assert online_bad()=='PRIVATE_ONLINE_RESULT'
        online_record=api('observability/traces/'+online_trace.receipt['id'])
        assert online_record['spans'][1]['usage']['evaluation_sampling']==dict(method='sha256_v1',sample_rate=1.0,selected=True)
        assert online_record['spans'][2]['usage']['evaluation_scores']=={'exact':1.0}
        assert online_record['spans'][2]['usage']['evaluator_ref']==dict(id='exact-check',version=2)
        assert online_record['spans'][4]['usage']['evaluator_ref']==dict(id='invalid-check',version=3)
        assert online_record['spans'][2]['parent_id']==1 and online_record['spans'][2]['status']=='completed'
        assert online_record['spans'][3]['status']=='completed' and online_record['spans'][4]['status']=='failed' and 'evaluation_scores' not in online_record['spans'][4]['usage']
        assert len(online_trace.evaluation_errors)==1 and 'PRIVATE_ONLINE_RESULT' not in json.dumps(online_record)
        online_export=tracked_client.export_trace(online_trace.receipt['id'])
        assert 'evaluator_ref' in json.dumps(online_export) and 'exact-check' in json.dumps(online_export)
        assert 'evaluation_scores' in json.dumps(online_export) and 'PRIVATE_ONLINE_RESULT' not in json.dumps(online_export)
        def bad_online_usage(scores,kind='tool',status='completed'):
            reject('observability/external-traces',dict(project_id='default',correlation_id='invalid-online-evaluation',started_ms=1000,spans=[dict(parent_id=None,kind='agent',name='root',started_ms=1000,duration_ms=10,status='completed',usage={}),dict(parent_id=0,kind=kind,name='evaluation',started_ms=1000,duration_ms=1,status=status,usage=dict(evaluation_scores=scores))]))
        for scores in [{},{'value':True},{'value':1.1},{'PRIVATE TEXT':0.5}]:bad_online_usage(scores)
        bad_online_usage({'value':0.5},kind='model');bad_online_usage({'value':0.5},status='failed')
        for evaluator_reference in [dict(id='check',version=True),dict(id='check',version=0),dict(id='PRIVATE TEXT',version=1),dict(id='check',version=1,private='PRIVATE')]:
            reject('observability/external-traces',dict(project_id='default',correlation_id='invalid-evaluator-reference',started_ms=1000,spans=[dict(parent_id=None,kind='tool',name='evaluation',started_ms=1000,duration_ms=1,status='completed',usage=dict(evaluator_ref=evaluator_reference))]))
        reject('observability/external-traces',dict(project_id='default',correlation_id='wrong-evaluator-kind',started_ms=1000,spans=[dict(parent_id=None,kind='model',name='evaluation',started_ms=1000,duration_ms=1,status='completed',usage=dict(evaluator_ref=dict(id='check',version=1)))]))
        @tracked_client.track('skip-evaluation',evaluate=lambda _: (_ for _ in ()).throw(AssertionError('should not evaluate')),evaluation_sample_rate=0)
        def skipped_task():return 'PRIVATE_SKIPPED_OUTPUT'
        with tracked_client.trace('default','sampling-zero-fixture') as skipped_trace:assert skipped_task()=='PRIVATE_SKIPPED_OUTPUT'
        skipped_record=api('observability/traces/'+skipped_trace.receipt['id'])
        assert len(skipped_record['spans'])==2 and skipped_record['spans'][1]['usage']['evaluation_sampling']==dict(method='sha256_v1',sample_rate=0,selected=False)
        assert 'evaluation_sampling' in json.dumps(tracked_client.export_trace(skipped_trace.receipt['id'])) and 'PRIVATE_SKIPPED_OUTPUT' not in json.dumps(skipped_record)
        rejected_sampling=dict(project_id='default',correlation_id='sampling-invalid',started_ms=1000,spans=[dict(parent_id=None,kind='tool',name='task',started_ms=1000,duration_ms=1,status='completed',usage=dict(evaluation_sampling=dict(method='sha256_v1',sample_rate=0,selected=True)))])
        reject('observability/external-traces',rejected_sampling)
        callback_summary_scope=dict(since_ms=online_record['started_ms'],until_ms=skipped_record['started_ms'])
        callback_summary=tracked_client.callback_evaluation_summary('default',**callback_summary_scope)
        assert callback_summary['selected_tasks']==2 and callback_summary['skipped_tasks']==1 and callback_summary['completed_assessments']==1 and callback_summary['failed_assessments']==1
        assert callback_summary['metrics']==[dict(evaluator_id='exact-check',evaluator_version=2,metric='exact',count=1,mean=1.0,min=1.0,max=1.0)]
        assert callback_summary['assessment_source']=='caller_reported' and callback_summary['provider_calls']==0 and callback_summary['automatic_promotion']==False
        assert 'PRIVATE_ONLINE_RESULT' not in json.dumps(callback_summary)
        exported_summary=json.loads(tracked_client.export_callback_evaluation_summary('default',**callback_summary_scope))
        assert exported_summary==callback_summary
        summary_csv=tracked_client.export_callback_evaluation_summary('default',format='csv',**callback_summary_scope)
        summary_rows=list(csv.DictReader(io.StringIO(summary_csv)))
        assert len(summary_rows)==1 and summary_rows[0]['evaluator_id']=='exact-check' and summary_rows[0]['evaluator_version']=='2'
        assert float(summary_rows[0]['mean'])==1 and summary_rows[0]['selected_tasks']=='2' and summary_rows[0]['failed_assessments']=='1'
        assert summary_rows[0]['since_ms']==str(callback_summary_scope['since_ms']) and summary_rows[0]['until_ms']==str(callback_summary_scope['until_ms'])
        empty_rows=list(csv.DictReader(io.StringIO(tracked_client.export_callback_evaluation_summary('foreign-project',format='csv',**callback_summary_scope))))
        assert len(empty_rows)==1 and empty_rows[0]['project_id']=='foreign-project' and empty_rows[0]['metric']=='' and empty_rows[0]['trace_count']=='0'
        assert 'PRIVATE_ONLINE_RESULT' not in summary_csv and len(requests)==online_calls
        print('PASS HTTP callback summary JSON/CSV exports preserve scope, counters, evaluator versions and empty context without model calls',flush=True)
        assert callback_summary['evaluators']==[
            dict(evaluator_id='exact-check',evaluator_version=2,completed_assessments=1,failed_assessments=0),
            dict(evaluator_id='invalid-check',evaluator_version=3,completed_assessments=0,failed_assessments=1)]
        attempt_csv=tracked_client.export_callback_evaluator_attempts_csv('default',**callback_summary_scope)
        attempt_rows=list(csv.DictReader(io.StringIO(attempt_csv)))
        assert len(attempt_rows)==2 and attempt_rows[0]['evaluator_id']=='exact-check' and attempt_rows[0]['completed_assessments']=='1' and attempt_rows[0]['failed_assessments']=='0'
        assert attempt_rows[1]['evaluator_id']=='invalid-check' and attempt_rows[1]['evaluator_version']=='3' and attempt_rows[1]['completed_assessments']=='0' and attempt_rows[1]['failed_assessments']=='1'
        assert attempt_rows[0]['since_ms']==str(callback_summary_scope['since_ms']) and attempt_rows[1]['until_ms']==str(callback_summary_scope['until_ms'])
        assert 'PRIVATE_ONLINE_RESULT' not in attempt_csv and len(requests)==online_calls
        version_gate_requirement=dict(evaluator_id='exact-check',evaluator_version=2,metric='exact',min_count=1,min_mean=1,max_failed_assessments=0)
        version_gate=tracked_client.check_callback_evaluation_summary('default',[version_gate_requirement],**callback_summary_scope)
        assert version_gate['passed'] and version_gate['checks'][0]['evaluator_attempts']['failed_assessments']==0
        assert not tracked_client.check_callback_evaluation_summary('default',[version_gate_requirement],max_failed_assessments=0,**callback_summary_scope)['passed']
        gate_requirement=dict(evaluator_id='exact-check',evaluator_version=2,metric='exact',min_count=1,min_mean=1,min_score=1)
        callback_gate=tracked_client.check_callback_evaluation_summary('default',[gate_requirement],max_failed_assessments=1,**callback_summary_scope)
        assert callback_gate['passed'] and callback_gate['summary']==callback_summary and callback_gate['automatic_promotion'] is False
        callback_gate_failed=tracked_client.check_callback_evaluation_summary('default',[gate_requirement],max_failed_assessments=0,**callback_summary_scope)
        assert not callback_gate_failed['passed'] and callback_gate_failed['failure_check']['scope']=='entire_summary'
        missing_gate=tracked_client.check_callback_evaluation_summary('default',[dict(gate_requirement,evaluator_version=1)],**callback_summary_scope)
        assert not missing_gate['passed'] and missing_gate['checks'][0]['reasons']==['missing_metric_group']
        assert len(requests)==online_calls
        print('PASS HTTP callback CI gate exact version/threshold ties, global failed limit and missing-version failure without model calls',flush=True)
        import xml.etree.ElementTree as CallbackET
        callback_artifacts=root/'callback-ci-artifacts';callback_artifacts.mkdir()
        callback_requirements=callback_artifacts/'callback-ci-requirements.json';callback_requirements.write_text(json.dumps([gate_requirement]))
        callback_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--callback-requirements-file',str(callback_requirements),'--callback-since-ms',str(callback_summary_scope['since_ms']),'--callback-until-ms',str(callback_summary_scope['until_ms'])]
        callback_report=callback_artifacts/'callback-ci-pass.json';callback_junit=callback_artifacts/'callback-ci-pass.xml'
        callback_ci=subprocess.run(callback_cli+['--callback-max-failed-assessments','1','--report',str(callback_report),'--junit',str(callback_junit)],capture_output=True,text=True,timeout=10)
        assert callback_ci.returncode==0,callback_ci.stdout+callback_ci.stderr
        assert json.loads(callback_report.read_text())==callback_gate
        callback_xml=CallbackET.parse(callback_junit).getroot();assert callback_xml.get('tests')=='2' and callback_xml.get('failures')=='0' and callback_xml.get('errors')=='0'
        callback_failed_report=callback_artifacts/'callback-ci-fail.json';callback_failed_junit=callback_artifacts/'callback-ci-fail.xml'
        callback_ci_failed=subprocess.run(callback_cli+['--callback-max-failed-assessments','0','--report',str(callback_failed_report),'--junit',str(callback_failed_junit)],capture_output=True,text=True,timeout=10)
        assert callback_ci_failed.returncode==1 and json.loads(callback_failed_report.read_text())==callback_gate_failed
        assert CallbackET.parse(callback_failed_junit).getroot().get('failures')=='1'
        assert len(requests)==online_calls
        print('PASS HTTP callback CI command exit codes and exact JSON/JUnit evidence without provider calls',flush=True)
        assert tracked_client.callback_evaluation_summary('foreign-project',**callback_summary_scope)['metrics']==[]
        assert tracked_client.callback_evaluation_summary('default',since_ms=skipped_record['started_ms']+1,until_ms=skipped_record['started_ms']+1)['metrics']==[]
        reject('observability/evaluations/summary?project_id=default&since_ms=2&until_ms=1')
        reject('observability/evaluations/summary?project_id=default&unexpected=true')
        print('PASS HTTP callback summary metric identity/value, sampling/failure counts, project/time isolation and metadata-only report',flush=True)
        assert len(requests)==online_calls
        print('PASS HTTP online callback evaluation scores, nested source, failure isolation, metadata export and native admission without model calls',flush=True)
        metric_catalog=Studio(base).metrics()
        assert metric_catalog['max_metrics_per_run']==6
        definitions={metric['id']:metric for metric in metric_catalog['metrics']}
        assert set(definitions)=={'exact_match','contains_reference','json_valid','whitespace_token_f1','json_equals','character_bigram_f1'}
        assert definitions['json_equals']['reference_requirement']=='valid_json'
        assert definitions['whitespace_token_f1']['reference_requirement']=='nonempty_tokens'
        assert all(metric['provider_calls']==0 and metric['higher_is_better'] and metric['min_score']==0 and metric['max_score']==1 for metric in definitions.values())
        print('PASS native metric discovery through SDK and explicit reference requirements')
        judge_guards=dict(settings,guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=policy_hash,action='block'))
        before_guarded_judge=len(requests)
        reject('evaluation/judge',dict(settings=judge_guards,rubric='Judge clarity.',input='private-marker',output='Answer',reference=None))
        assert len(requests)==before_guarded_judge
        verdict_policy=Studio(base).save_guardrail_policy([dict(id='verdict-score',kind='forbidden_substrings',value=['score'])])['policy']['policy_sha256']
        judge_guards['guardrails']['output_policy_sha256']=verdict_policy
        reject('evaluation/judge',dict(settings=judge_guards,rubric='Judge clarity.',input='Question',output='Answer',reference=None))
        assert len(requests)==before_guarded_judge+1
        judge_guards['guardrails']['action']='observe'
        observed_judge=api('evaluation/judge',dict(settings=judge_guards,rubric='Judge clarity.',input='Question',output='Answer',reference=None))
        assert observed_judge['score']==0.75
        observed_judge_trace=api('observability/traces/'+observed_judge['trace_id'])
        assert any(span['usage'].get('guardrail_receipt',{}).get('passed') is False for span in observed_judge_trace['spans'])
        print('PASS judge policies block input before inference, withhold blocked verdict and retain observed verdict with per-rule trace',flush=True)
        judged=api('evaluation/judge',dict(settings=settings,rubric='Judge clarity.',input='Question',output='Answer',reference=None))
        assert judged['score']==0.75 and judged['automatic_promotion'] is False and len(judged['source_sha256'])==64
        assert Studio(base).judgment(judged['id'])==judged
        judge_path=data/'evaluation'/'judges'/f"{judged['id']}.json"
        original_judge=judge_path.read_bytes();changed=json.loads(original_judge);changed['score']=1
        judge_path.write_text(json.dumps(changed));reject('evaluation/judge/'+judged['id']);judge_path.write_bytes(original_judge)
        sdk_judge=Studio(base).judge(settings,'Judge clarity.','Question','Answer')
        assert sdk_judge['score']==0.75 and Studio(base).judgment(sdk_judge['id'])==sdk_judge
        trace_judge_calls=len(requests)
        trace_judged=Studio(base).judge_trace(external_receipt['id'],online_selection['trace_sha256'],settings,'Judge clarity.','Explicit question','Explicit answer')
        assert trace_judged['score']==0.75 and trace_judged['trace_ref']==dict(trace_id=external_receipt['id'],trace_sha256=online_selection['trace_sha256'])
        assert trace_judged['answer_source']=='caller_supplied' and trace_judged['trace_content_verified'] is False
        assert Studio(base).judgment(trace_judged['id'])==trace_judged and len(requests)==trace_judge_calls+1
        reject('evaluation/judge',dict(settings=settings,rubric='Judge clarity.',input='Question',output='Answer',trace_ref=dict(trace_id=external_receipt['id'],trace_sha256='f'*64)))
        assert len(requests)==trace_judge_calls+1
        source_calls=len(requests)
        quality_source=Studio(base).save_online_quality_source('default',external_receipt['id'],online_selection['trace_sha256'],'Pinned quality question','Pinned quality answer')
        quality_judged=Studio(base).judge_online_quality_source(quality_source['source_sha256'],settings,'Judge clarity.')
        assert quality_judged['input']=='Pinned quality question' and quality_judged['output']=='Pinned quality answer'
        assert quality_judged['quality_source_sha256']==quality_source['source_sha256'] and quality_judged['trace_ref']==trace_judged['trace_ref']
        assert quality_judged['answer_source']=='caller_supplied' and quality_judged['trace_content_verified'] is False
        assert Studio(base).judgment(quality_judged['id'])==quality_judged and len(requests)==source_calls+1
        reject('evaluation/judge',dict(settings=settings,rubric='Judge clarity.',input='forged',output='Pinned quality answer',trace_ref=quality_judged['trace_ref'],quality_source_ref=quality_source['source_sha256']))
        assert len(requests)==source_calls+1
        quality_job=Studio(base).submit_online_quality_job(quality_source['source_sha256'],settings,'Judge clarity.')
        assert Studio(base).submit_online_quality_job(quality_source['source_sha256'],settings,'Judge clarity.')['id']==quality_job['id']
        until=time.time()+10
        while time.time()<until:
            quality_job=Studio(base).online_quality_job(quality_job['id'])
            if quality_job['status'] in ['completed','failed','interrupted']:break
            time.sleep(.05)
        assert quality_job['status']=='completed' and quality_job['result']['automatic_execution'] is True
        assert len(requests)==source_calls+2
        background_judge=Studio(base).judgment(quality_job['result']['judge_id'])
        assert background_judge['quality_source_sha256']==quality_source['source_sha256'] and background_judge['receipt_sha256']==quality_job['result']['judge_receipt_sha256']
        assert Studio(base).submit_online_quality_job(quality_source['source_sha256'],settings,'Judge clarity.')==quality_job
        assert len(requests)==source_calls+2
        model_rule_calls=len(requests);model_config=Studio(base).save_online_model_evaluator(settings,'Judge clarity.')
        assert Studio(base).online_model_evaluator('default',model_config['evaluator_sha256'])==model_config
        model_rule=Studio(base).save_online_evaluation_rule(dict(id='model-answer-quality',project_id='default',evaluator_id=model_config['evaluator_id'],evaluator_version=1,sample_rate=1.0,enabled=True))
        Studio(base).bind_online_evaluation_rule(model_rule['rule_sha256'],base_version=0,active=True)
        model_trace=Studio(base).ingest_trace('default','automatic-model-source',1000,[dict(parent_id=None,kind='agent',name='root',status='completed',started_ms=1000,duration_ms=1)])
        model_selection=Studio(base).select_online_evaluations('default',model_trace['id'])
        assert len(requests)==model_rule_calls
        automatic_source=Studio(base).save_online_quality_source('default',model_trace['id'],model_selection['trace_sha256'],'Automatic question','Automatic answer')
        automatic_job=None;until=time.time()+10
        while time.time()<until:
            automatic_job=next((row for row in Studio(base).online_quality_jobs('default')['jobs'] if row['source_sha256']==automatic_source['source_sha256']),None)
            if automatic_job and automatic_job['status'] in ['completed','failed','interrupted']:break
            time.sleep(.05)
        assert automatic_job['status']=='completed' and len(requests)==model_rule_calls+1
        automatic_status=Studio(base).online_quality_job(automatic_job['id'])
        assert automatic_status['plan']['request']['rule_pin']['evaluator_sha256']==model_config['evaluator_sha256']
        assert automatic_status['plan']['request']['rule_pin']['selection_sha256']==model_selection['selection_sha256']
        Studio(base).bind_online_evaluation_rule(model_rule['rule_sha256'],base_version=1,active=False)
        assert Studio(base).save_online_quality_source('default',model_trace['id'],model_selection['trace_sha256'],'Automatic question','Automatic answer')==automatic_source
        assert len(requests)==model_rule_calls+1
        online_catalog=online_client.online_evaluation_rules('default',limit=1);online_catalog_next=online_client.online_evaluation_rules('default',offset=1,limit=1)

        assert json.loads((data/'evaluation'/'judges'/f"{judged['id']}.json").read_text())==judged
        judge_trace=trace_rows('judge-'+judged['id'])[0]
        assert judge_trace['id']==judged['trace_id'] and judge_trace['status']=='completed' and len(judge_trace['spans'])==2
        assert 'Judge clarity' not in json.dumps(judge_trace) and judged['reason'] not in json.dumps(judge_trace)
        reject('evaluation/judge',dict(settings=settings,rubric='BAD_JUDGE',input='Question',output='Answer'))
        before_calls=len(requests)
        reject('evaluation/judge',dict(settings=settings,rubric='',input='Question',output='Answer'))
        assert len(requests)==before_calls
        print('PASS explicit bounded rubric judge, immutable evidence, metadata-only native trace and invalid verdict rejection')

        dataset=api('evaluation/datasets',dict(project_id='default',name='Cases',base_version=0,samples=[dict(id='a',input='good',expected_output='answer: good',metadata={'source':'fixture'})]))
        revision=api('evaluation/datasets',dict(id=dataset['id'],project_id='default',name='Cases',base_version=1,samples=[dict(id='a',input='changed',expected_output='answer: changed')]))
        assert revision['version']==2 and revision['sha256']!=dataset['sha256']
        versions=Studio(base).dataset_versions(dataset['id'],'default',limit=1);assert versions['total']==2 and versions['versions'][0]['sha256']==revision['sha256'] and versions['has_more']
        assert Studio(base).dataset_versions(dataset['id'],'default',offset=1,limit=1)['versions'][0]['sha256']==dataset['sha256']
        reject('evaluation/datasets/'+dataset['id']+'/versions?project_id=default&limit=101')
        before_compare=len(requests)
        comparison=Studio(base).compare_dataset_versions(dataset['id'],'default',1,2,limit=1)
        assert comparison['from']['sha256']==dataset['sha256'] and comparison['to']['sha256']==revision['sha256']
        assert comparison['counts']==dict(added=0,removed=0,changed=1,unchanged=0)
        assert comparison['changes']==[dict(sample_id='a',kind='changed',fields=['input','expected_output','metadata'])]
        dataset_csv=list(csv.DictReader(io.StringIO(Studio(base).export_dataset_comparison_csv(dataset['id'],'default',1,2))))
        assert len(dataset_csv)==1 and dataset_csv[0]['sample_id']=='a' and dataset_csv[0]['fields']=='input;expected_output;metadata'
        assert dataset_csv[0]['from_sha256']==dataset['sha256'] and dataset_csv[0]['to_sha256']==revision['sha256']
        assert Studio(base).compare_dataset_versions(dataset['id'],'default',2,2)['total']==0
        reject('evaluation/datasets/'+dataset['id']+'/compare?project_id=default&from_version=1&to_version=2&limit=101')
        assert len(requests)==before_compare
        print('PASS immutable dataset version comparison pins hashes and changed fields without provider calls',flush=True)
        before_fork=len(requests)
        variant=Studio(base).fork_dataset(dataset['id'],1,'default','Cases variant')
        assert variant['id']!=dataset['id'] and variant['version']==1 and variant['samples']==dataset['samples']
        assert variant['origin']==dict(id=dataset['id'],version=1,sha256=dataset['sha256'])
        variant_next=api('evaluation/datasets',dict(id=variant['id'],project_id='default',name='Changed variant',base_version=1,samples=variant['samples']))
        assert variant_next['origin']==variant['origin']
        reject('evaluation/datasets',dict(project_id='default',name='Bad origin',base_version=0,samples=dataset['samples'],origin=dict(id=dataset['id'],version=1,sha256='0'*64)))
        assert api('evaluation/datasets/'+dataset['id']+'/versions/1')==dataset and len(requests)==before_fork
        print('PASS dataset variants pin original revision/hash, retain ancestry across edits and leave source unchanged without inference',flush=True)
        archive_data=api('evaluation/datasets',dict(project_id='default',name='Archive fixture',base_version=0,samples=[dict(id='a',input='retained')]))
        archive_client=Studio(base);before_archive=len(requests)
        archived=archive_client.dataset_lifecycle(archive_data['id'],'default',base_version=1,base_revision=0,archived=True)
        assert archived['lifecycle_revision']==1 and archived['snapshots_retained']
        assert all(row['id']!=archive_data['id'] for row in archive_client.datasets('default')['datasets'])
        assert any(row['id']==archive_data['id'] for row in archive_client.datasets('default',archived=True)['datasets'])
        assert api('evaluation/datasets/'+archive_data['id']+'/versions/1')==archive_data
        reject('evaluation/datasets',dict(id=archive_data['id'],project_id='default',name='changed',base_version=1,samples=[dict(id='a',input='changed')]))
        try:archive_client.dataset_lifecycle(archive_data['id'],'default',base_version=1,base_revision=0,archived=False);raise AssertionError('stale archive restore accepted')
        except EvaluationError as error:assert error.reason=='http_409'
        archive_client.dataset_lifecycle(archive_data['id'],'default',base_version=1,base_revision=1,archived=False)
        assert any(row['id']==archive_data['id'] and row['lifecycle_revision']==2 for row in archive_client.datasets('default')['datasets']) and len(requests)==before_archive
        active_catalog=archive_client.datasets('default');catalog_page=archive_client.datasets('default',offset=0,limit=1)
        assert catalog_page['total']==len(active_catalog['datasets']) and catalog_page['datasets']==active_catalog['datasets'][:1]
        reject('evaluation/datasets?project_id=default&limit=101')
        searched=archive_client.datasets('default',search='ARCHIVE FIXTURE',limit=20);assert searched['total']==1 and searched['datasets'][0]['id']==archive_data['id']
        assert archive_client.datasets('default',search=archive_data['id'])['total']==1
        reject('evaluation/datasets?'+urllib.parse.urlencode(dict(project_id='default',q='x'*201)))
        print('PASS recoverable dataset archive hides active catalog, retains immutable versions, blocks edits and rejects stale restore without inference',flush=True)
        pinned_judge=Studio(base).judge_sample(dataset['id'],1,'a',settings,'Judge correctness.','answer: good')
        assert pinned_judge['dataset_ref']==dict(dataset_id=dataset['id'],dataset_version=1,dataset_sha256=dataset['sha256'],sample_id='a')
        assert pinned_judge['input']=='good' and pinned_judge['reference']=='answer: good'
        before_calls=len(requests)
        reject('evaluation/judge',dict(settings=settings,rubric='Judge correctness.',input='changed',output='answer',reference='answer: good',dataset_ref=dict(dataset_id=dataset['id'],dataset_version=1,sample_id='a')))
        assert len(requests)==before_calls
        print('PASS model judge pins dataset version, hash and sample despite newer revisions; mismatched input rejects before provider')
        judge_data=api('evaluation/datasets',dict(project_id='default',name='Batch judge',base_version=0,samples=[dict(id='a',input='Question one'),dict(id='b',input='Question two')]))
        before_plan=len(requests)
        judge_plan=Studio(base).plan_judges(judge_data['id'],1,{'a':'Answer','b':'Answer'},settings,'Judge clarity.')
        assert judge_plan['provider_calls']==0 and judge_plan['dataset_sha256']==judge_data['sha256'] and len(judge_plan['samples'])==2
        assert Studio(base).judge_plan(judge_plan['id'])==judge_plan
        api('evaluation/datasets',dict(id=judge_data['id'],project_id='default',name='Batch judge revised',base_version=1,samples=[dict(id='a',input='Changed')]))
        assert Studio(base).judge_plan(judge_plan['id'])==judge_plan
        plan_path=data/'evaluation'/'judge_plans'/f"{judge_plan['id']}.json"
        original_plan=plan_path.read_bytes();changed_plan=json.loads(original_plan);changed_plan['rubric']='Changed criterion'
        plan_path.write_text(json.dumps(changed_plan));reject('evaluation/judge-plans/'+judge_plan['id']);plan_path.write_bytes(original_plan)
        reject('evaluation/judge-plans',dict(dataset_id=judge_data['id'],dataset_version=1,outputs={'a':'Answer'},settings=settings,rubric='Judge clarity.'))
        assert len(requests)==before_plan
        print('PASS frozen validated judge plans, hash checks and version pinning without provider invocation')
        rubric_version=Studio(base).save_rubric('default','Clarity criterion','Judge clarity.')
        before_rubric_pin=len(requests)
        rubric_plan=Studio(base).plan_judges(judge_data['id'],1,{'a':'Answer','b':'Answer'},settings,rubric_ref=dict(id=rubric_version['id'],version=1))
        assert rubric_plan['rubric']=='Judge clarity.' and rubric_plan['rubric_snapshot']==rubric_version
        Studio(base).save_rubric('default','Clarity criterion','Changed criterion.',rubric_id=rubric_version['id'],base_version=1)
        assert Studio(base).judge_plan(rubric_plan['id'])==rubric_plan
        reject('evaluation/judge-plans',dict(dataset_id=judge_data['id'],dataset_version=1,outputs={'a':'Answer','b':'Answer'},settings=settings,rubric='Inline',rubric_ref=dict(id=rubric_version['id'],version=1)))
        reject('evaluation/judge-plans',dict(dataset_id=judge_data['id'],dataset_version=1,outputs={'a':'Answer','b':'Answer'},settings=settings,rubric_ref=dict(id=rubric_version['id'],version=0)))
        assert len(requests)==before_rubric_pin
        print('PASS versioned rubric snapshot pins old criterion across edits and rejects ambiguous/latest references without provider calls')
        def wait_judges(run_id,status=None):
            until=time.time()+8
            while time.time()<until:
                run=Studio(base).judge_run(run_id)
                if run['status']!='running' and (status is None or run['status']==status):return run
                time.sleep(.03)
            raise AssertionError(run)
        native_judges=Studio(base).start_judges(judge_plan['id'])['id']
        native_completed=wait_judges(native_judges,'completed')
        pinned_rubric_run=Studio(base).start_judges(rubric_plan['id'])['id']
        pinned_rubric_result=wait_judges(pinned_rubric_run,'completed')
        assert all(Studio(base).judgment(item['receipt_id'])['rubric']=='Judge clarity.' for item in pinned_rubric_result['items'])
        preset_catalog=Studio(base).judge_presets()
        presets={preset['id']:preset for preset in preset_catalog['presets']}
        assert set(presets)=={'answer_relevance','reference_correctness','source_faithfulness'} and preset_catalog['provider_calls']==0
        grounded_data=api('evaluation/datasets',dict(project_id='default',name='Grounded fixture',base_version=0,samples=[dict(id='source',input='What does the evidence say?',contexts=['The fixture value is 42.'],expected_output='42')]))
        before_presets=len(requests)
        grounded_plans={}
        for preset_id in presets:
            grounded_plans[preset_id]=Studio(base).plan_judges(grounded_data['id'],1,{'source':'42'},settings,judge_preset=dict(id=preset_id,version=1))
            assert grounded_plans[preset_id]['judge_preset']==presets[preset_id]
            assert Studio(base).judge_plan(grounded_plans[preset_id]['id'])==grounded_plans[preset_id]
        for preset_id in ['reference_correctness','source_faithfulness']:
            reject('evaluation/judge-plans',dict(dataset_id=judge_data['id'],dataset_version=1,outputs={'a':'Answer','b':'Answer'},settings=settings,judge_preset=dict(id=preset_id,version=1)))
        reject('evaluation/judge-plans',dict(dataset_id=grounded_data['id'],dataset_version=1,outputs={'source':'42'},settings=settings,judge_preset=dict(id='source_faithfulness',version=2)))
        assert len(requests)==before_presets
        grounded_run=Studio(base).start_judges(grounded_plans['source_faithfulness']['id'])['id']
        grounded_result=wait_judges(grounded_run,'completed')
        grounded_request=json.loads(requests[-1]['messages'][-1]['content'])
        assert grounded_request['contexts']==['The fixture value is 42.'] and grounded_request['rubric']==presets['source_faithfulness']['rubric']
        assert Studio(base).judgment(grounded_result['items'][0]['receipt_id'])['rubric']==presets['source_faithfulness']['rubric']
        print('PASS version-pinned model judge presets, evidence preflight and actual context/rubric delivery; scoring quality remains unqualified')
        assert len(native_completed['items'])==2 and native_completed['mean_score']==0.75
        before_catalog=len(requests)
        catalog=Studio(base).judge_runs('default',limit=1)
        assert catalog['total']>=1 and len(catalog['runs'])==1
        assert any(row['id']==native_judges and row['mean_score']==0.75 and row['completed']==2 for row in Studio(base).judge_runs('default')['runs'])
        assert Studio(base).judge_runs('absent')['total']==0
        assert Studio(base).judge_runs('default',offset=1000)['runs']==[]
        reject('evaluation/judge-runs?project_id=default&limit=21')
        assert len(requests)==before_catalog
        batch_trace=trace_rows('judge-run-'+native_judges)[0]
        assert batch_trace['id']==native_completed['trace_id'] and batch_trace['status']=='completed'
        assert [span['kind'] for span in batch_trace['spans']]==['judge_batch','judge_item','judge_item']
        assert_trace_tree(batch_trace)
        assert [span['linked_trace_id'] for span in batch_trace['spans'][1:]]==[item['trace_id'] for item in native_completed['items']]
        assert all(span['usage']=={} for span in batch_trace['spans'])
        assert api('observability/traces/'+batch_trace['id'])==batch_trace
        for span in batch_trace['spans'][1:]:
            linked=api('observability/traces/'+span['linked_trace_id'])
            assert linked['project_id']==batch_trace['project_id'] and linked['status']=='completed'
        reject('observability/traces/missing')

        assert 'Judge clarity.' not in json.dumps(batch_trace) and 'Answer' not in json.dumps(batch_trace)

        cancel_plan=Studio(base).plan_judges(judge_data['id'],1,{'a':'Answer','b':'SLOW_JUDGE'},settings,'Judge clarity.')
        cancelled_judges=Studio(base).start_judges(cancel_plan['id'])['id']
        until=time.time()+5
        while time.time()<until:
            state=Studio(base).judge_run(cancelled_judges)
            if len(state['items'])==1:break
            time.sleep(.03)
        assert len(state['items'])==1
        Studio(base).cancel_judges(cancelled_judges)
        cancelled_result=wait_judges(cancelled_judges,'cancelled')
        assert len(cancelled_result['items'])==1 and cancelled_result['mean_score'] is None
        high_plan=Studio(base).plan_judges(judge_data['id'],1,{'a':'JUDGE_SCORE_HIGH','b':'Answer'},settings,'Judge clarity.')
        high_run=Studio(base).start_judges(high_plan['id'])['id'];wait_judges(high_run,'completed')
        mixed_plan=Studio(base).plan_judges(judge_data['id'],1,{'a':'JUDGE_SCORE_HIGH','b':'JUDGE_SCORE_LOW'},settings,'Judge clarity.')
        mixed_run=Studio(base).start_judges(mixed_plan['id'])['id'];mixed=wait_judges(mixed_run,'completed')
        before_compare=len(requests)
        improvement=Studio(base).compare_judges(native_judges,high_run)
        assert improvement['eligible'] and improvement['improvements']==1 and improvement['regressions']==0
        regression=Studio(base).compare_judges(native_judges,mixed_run)
        assert mixed['mean_score']>native_completed['mean_score'] and not regression['eligible'] and regression['regressions']==1 and regression['reason']=='paired_regression'
        tie=Studio(base).compare_judges(native_judges,native_judges)
        reject('evaluation/judge-runs/compare',dict(baseline_id=native_judges,candidate_id=pinned_rubric_run))
        assert not tie['eligible'] and tie['reason']=='no_improvement'
        reject('evaluation/judge-runs/compare',dict(baseline_id=native_judges,candidate_id=cancelled_judges))
        assert len(requests)==before_compare
        assert json.loads((data/'evaluation'/'judge_comparisons'/f"{regression['id']}.json").read_text())==regression
        print('PASS persisted paired judge comparison blocks per-example regressions despite higher mean, rejects partial runs and never invokes provider')
        cancelled_batch_trace=trace_rows('judge-run-'+cancelled_judges)[0]
        assert cancelled_batch_trace['status']=='interrupted' and not any(span['status']=='running' for span in cancelled_batch_trace['spans'])

        sdk_batch=Studio(base).evaluate_judge_plan(judge_plan['id'],min_score=0.7,poll_interval=.02)
        assert sdk_batch['passed'] and sdk_batch['observed']==0.75
        sdk_failed=Studio(base).evaluate_judge_plan(judge_plan['id'],min_score=0.8,poll_interval=.02)
        assert not sdk_failed['passed'] and sdk_failed['run']['status']=='completed'
        try:
            Studio(base).evaluate_judge_plan(cancel_plan['id'],timeout=.2,poll_interval=.02)
            raise AssertionError('Slow judge did not time out')
        except EvaluationError as error:
            assert error.reason=='judge_timeout' and error.run_id and not error.cancellation_failed
            timed_judges=wait_judges(error.run_id,'cancelled')
            assert timed_judges['mean_score'] is None
        judge_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--judge-plan-id',judge_plan['id']]
        judge_report=root/'judge-report.json';judge_junit=root/'judge-failed.xml'
        judge_ci_good=subprocess.run(judge_cli+['--judge-min-score','0.7','--report',str(judge_report)],capture_output=True,text=True,timeout=10)
        assert judge_ci_good.returncode==0 and json.loads(judge_ci_good.stdout)['mean_score']==0.75
        assert json.loads(judge_report.read_text())['run']['plan_sha256']==judge_plan['plan_sha256']
        judge_ci_bad=subprocess.run(judge_cli+['--judge-min-score','0.8','--junit',str(judge_junit)],capture_output=True,text=True,timeout=10)
        assert judge_ci_bad.returncode==1 and not json.loads(judge_ci_bad.stdout)['passed']
        import xml.etree.ElementTree as ET
        assert ET.parse(judge_junit).getroot().get('failures')=='1'
        judge_ci_timeout=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--judge-plan-id',cancel_plan['id'],'--timeout','.2'],capture_output=True,text=True,timeout=10)
        timeout_summary=json.loads(judge_ci_timeout.stdout)
        assert judge_ci_timeout.returncode==2 and timeout_summary['run_id'] and not timeout_summary['cancellation_failed']
        assert wait_judges(timeout_summary['run_id'],'cancelled')['mean_score'] is None
        print('PASS judge SDK and native CI thresholds, JSON/JUnit evidence and timeout cancellation')
        judge_ci_tie=subprocess.run(judge_cli+['--baseline-id',native_judges],capture_output=True,text=True,timeout=10)
        assert judge_ci_tie.returncode==0 and json.loads(judge_ci_tie.stdout)['regressions']==0
        judge_ci_strict=subprocess.run(judge_cli+['--baseline-id',native_judges,'--require-improvement'],capture_output=True,text=True,timeout=10)
        assert judge_ci_strict.returncode==1 and json.loads(judge_ci_strict.stdout)['improvements']==0
        judge_ci_regression=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--judge-plan-id',mixed_plan['id'],'--baseline-id',native_judges],capture_output=True,text=True,timeout=10)
        assert judge_ci_regression.returncode==1 and json.loads(judge_ci_regression.stdout)['mean_score']>0.75 and json.loads(judge_ci_regression.stdout)['regressions']==1
        judge_ci_gain=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--judge-plan-id',high_plan['id'],'--baseline-id',native_judges,'--require-improvement'],capture_output=True,text=True,timeout=10)
        assert judge_ci_gain.returncode==0 and json.loads(judge_ci_gain.stdout)['improvements']==1
        mismatched_plan=Studio(base).plan_judges(judge_data['id'],1,{'a':'Answer','b':'Answer'},settings,'Other criterion.')
        before_preflight=len(requests)
        for plan_id,baseline_id in [(mismatched_plan['id'],native_judges),(judge_plan['id'],cancelled_judges)]:
            try:Studio(base).evaluate_judge_plan(plan_id,baseline_id=baseline_id)
            except EvaluationError:pass
            else:raise AssertionError('Invalid judge baseline accepted')
        assert len(requests)==before_preflight
        print('PASS judge CI paired baseline, strict ties, improvement and regression exits, incompatible preflight without model calls')

        judged_batch=Studio(base).judge_outputs(judge_data['id'],1,{'a':'Answer','b':'Answer'},settings,'Judge clarity.',min_score=0.7)
        assert judged_batch['passed'] and judged_batch['mean_score']==0.75 and len(judged_batch['items'])==2
        failed_threshold=Studio(base).judge_outputs(judge_data['id'],1,{'a':'Answer','b':'Answer'},settings,'Judge clarity.',min_score=0.8)
        assert not failed_threshold['passed']
        before_batch=len(requests)
        try:
            Studio(base).judge_outputs(judge_data['id'],1,{'a':'Answer','b':'BAD_JUDGE'},settings,'Judge clarity.')
            raise AssertionError('Invalid verdict accepted')
        except EvaluationError as error:
            assert error.sample_id=='b' and len(error.completed)==1 and len(error.receipt_ids)==1
            assert Studio(base).judgment(error.receipt_ids[0])==error.completed[0]['receipt']
        assert len(requests)==before_batch+2
        print('PASS batch rubric judge thresholds and retained partial receipts without replay after failure')


        assert api(f"evaluation/datasets/{dataset['id']}/versions/1")['samples'][0]['input']=='good'
        try:api('evaluation/datasets',dict(id=dataset['id'],project_id='default',name='Stale',base_version=1,samples=[dict(id='a',input='wrong')]));raise AssertionError('stale dataset update accepted')
        except urllib.error.HTTPError as e:assert e.code==409
        def experiment(snapshot,model='mock',template='{{input}}'):
            return api('evaluation/experiments',dict(dataset_id=snapshot['id'],dataset_version=snapshot['version'],settings=dict(settings,model=model),prompt_template=template,metrics=['exact_match'],concurrency=2))['id']
        def wait_run(run_id):
            until=time.time()+10
            while time.time()<until:
                run=api('evaluation/experiments/'+run_id)
                if run['status']!='running':return run
                time.sleep(.03)
            raise AssertionError(run)
        guarded_eval_data=api('evaluation/datasets',dict(project_id='default',name='Guarded native evaluation',base_version=0,samples=[dict(id='blocked',input='private-marker',expected_output='Answer')]))
        guarded_eval_settings=dict(settings,guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=policy_hash,action='block'))
        before_guarded_eval=len(requests)
        input_blocked_eval=wait_run(api('evaluation/experiments',dict(dataset_id=guarded_eval_data['id'],dataset_version=1,settings=guarded_eval_settings,prompt_template='{{input}}',metrics=['exact_match']))['id'])
        assert len(requests)==before_guarded_eval and input_blocked_eval['items'][0]['error']=='guardrail_blocked'
        assert input_blocked_eval['items'][0]['output'] is None and not input_blocked_eval['items'][0]['scores']
        output_eval_policy=Studio(base).save_guardrail_policy([dict(id='no-answer',kind='forbidden_substrings',value=['answer:'])])['policy']['policy_sha256']
        guarded_eval_data=api('evaluation/datasets',dict(project_id='default',name='Guarded safe evaluation',base_version=0,samples=[dict(id='safe',input='safe',expected_output='answer: safe')]))
        guarded_eval_settings['guardrails']['output_policy_sha256']=output_eval_policy
        output_blocked_eval=wait_run(api('evaluation/experiments',dict(dataset_id=guarded_eval_data['id'],dataset_version=1,settings=guarded_eval_settings,prompt_template='{{input}}',metrics=['exact_match']))['id'])
        assert len(requests)==before_guarded_eval+1 and output_blocked_eval['items'][0]['error']=='guardrail_blocked'
        assert output_blocked_eval['items'][0]['output'] is None and not output_blocked_eval['items'][0]['scores']
        assert output_blocked_eval['items'][0]['usage']['completion_tokens']>0
        guarded_eval_settings['guardrails']['action']='observe'
        observed_eval=wait_run(api('evaluation/experiments',dict(dataset_id=guarded_eval_data['id'],dataset_version=1,settings=guarded_eval_settings,prompt_template='{{input}}',metrics=['exact_match']))['id'])
        assert observed_eval['items'][0]['status']=='completed' and observed_eval['items'][0]['output']=='answer: safe'
        guard_eval_trace=api('observability/traces/'+output_blocked_eval['trace_id'])
        assert any(span['usage'].get('guardrail_receipt',{}).get('blocked') for span in guard_eval_trace['spans'])
        print('PASS native evaluation policies input no inference, blocked output without scores with usage, observe and per-rule trace',flush=True)
        bigram_data=api('evaluation/datasets',dict(project_id='default',name='Character order',base_version=0,samples=[dict(id='a',input='order',expected_output='aaaa',contexts=['reference source'])]))
        bigram_score=api('evaluation/score',dict(dataset_id=bigram_data['id'],dataset_version=1,metrics=['character_bigram_f1'],outputs={'a':'aa'}))
        assert bigram_score['items'][0]['scores']['character_bigram_f1']==0.5
        bigram_baseline=api('evaluation/score',dict(dataset_id=bigram_data['id'],dataset_version=1,metrics=['character_bigram_f1'],outputs={'a':'aaaa'}))
        bigram_comparison=api('evaluation/score/compare',dict(baseline_id=bigram_baseline['id'],candidate_id=bigram_score['id']))
        assert bigram_comparison['regressions']==1 and not bigram_comparison['eligible']
        assert bigram_baseline['provider_calls']==bigram_score['provider_calls']==0
        task_calls=[]
        task_result=Studio(base).evaluate_task(bigram_data['id'],1,lambda text:task_calls.append(text) or 'aaaa',['character_bigram_f1'],min_scores={'character_bigram_f1':1})
        assert task_calls==['order'] and task_result['passed']
        assert task_result['receipt']['mean_scores']['character_bigram_f1']==1 and task_result['receipt']['provider_calls']==0
        async_task_calls=[]
        async def async_task(text):
            async_task_calls.append(text)
            return 'aa'
        async_task_result=__import__('asyncio').run(Studio(base).evaluate_task_async(bigram_data['id'],1,async_task,['character_bigram_f1'],baseline_id=task_result['receipt']['id']))
        assert async_task_calls==['order'] and not async_task_result['passed']
        assert async_task_result['receipt']['mean_scores']['character_bigram_f1']==0.5
        assert async_task_result['comparison']['regressions']==1 and async_task_result['receipt']['provider_calls']==0
        rag_inputs=[]
        def rag_task(text,contexts):
            rag_inputs.append((text,contexts))
            return 'aaaa'
        rag_result=Studio(base).evaluate_task(bigram_data['id'],1,rag_task,['character_bigram_f1'],with_contexts=True)
        assert rag_inputs==[('order',('reference source',))] and rag_result['passed']
        assert rag_result['receipt']['provider_calls']==0
        parallel_data=api('evaluation/datasets',dict(project_id='default',name='Parallel Python task',base_version=0,samples=[dict(id=str(i),input=str(i),expected_output='ok') for i in range(4)]))
        parallel_calls=[]
        async def parallel_task(text):
            parallel_calls.append(text)
            await __import__('asyncio').sleep(0)
            return 'ok'
        parallel_result=__import__('asyncio').run(Studio(base).evaluate_task_async(parallel_data['id'],1,parallel_task,['exact_match'],concurrency=2))
        assert len(parallel_calls)==4 and len(set(parallel_calls))==4 and parallel_result['passed']
        assert parallel_result['receipt']['mean_scores']=={'exact_match':1} and parallel_result['receipt']['provider_calls']==0
        traced_client=Studio(base)
        with traced_client.trace('default','python-task-evaluation') as task_trace:
            traced_result=__import__('asyncio').run(traced_client.evaluate_task_async(parallel_data['id'],1,parallel_task,['exact_match'],concurrency=2,trace=task_trace))
        traced_record=api('observability/traces/'+task_trace.receipt['id'])
        assert len(traced_record['spans'])==5 and traced_result['passed']
        assert all(span['status']=='completed' for span in traced_record['spans'])
        assert [span['name'] for span in traced_record['spans'][1:]]==['sample.'+str(i) for i in range(4)]
        assert all(span['parent_id']==traced_record['spans'][0]['id'] for span in traced_record['spans'][1:])
        task_source=root/'ci-task.py'
        task_source.write_text("def task(text,contexts):\n    assert contexts == ('reference source',)\n    return 'aaaa'\nasync def async_task(text,contexts):\n    return 'aa'\n")
        task_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--dataset-id',bigram_data['id'],'--dataset-version','1','--task-file',str(task_source),'--with-contexts','--metric','character_bigram_f1']
        task_report=root/'task-report.json';task_junit=root/'task-report.xml'
        task_ci=subprocess.run(task_cli+['--trace-correlation','ci-task-trace','--trace-key','ci-task-key','--min-score','character_bigram_f1=1','--report',str(task_report),'--junit',str(task_junit)],capture_output=True,text=True,timeout=10)
        assert task_ci.returncode==0,task_ci.stderr+task_ci.stdout
        assert json.loads(task_report.read_text())['receipt']['mean_scores']=={'character_bigram_f1':1}
        traced_ci_report=json.loads(task_report.read_text())
        assert traced_ci_report['trace_id'] and not traced_ci_report['trace_export_failed']
        traced_ci_record=api('observability/traces/'+traced_ci_report['trace_id'])
        assert [span['name'] for span in traced_ci_record['spans']]==['pipeline','sample.a']
        assert all(span['status']=='completed' for span in traced_ci_record['spans'])
        task_xml=ET.parse(task_junit).getroot()
        assert task_xml.attrib['tests']=='2' and task_xml.attrib['failures']=='0'
        assert [case.get('name') for case in task_xml.findall('testcase')]==['a','quality_gate']
        task_ci_async=subprocess.run(task_cli+['--task-function','async_task','--task-async','--min-score','character_bigram_f1=1'],capture_output=True,text=True,timeout=10)
        assert task_ci_async.returncode==1 and json.loads(task_ci_async.stdout)['mean_scores']=={'character_bigram_f1':0.5}
        task_source.write_text("def task(text):\n    if text == '1': raise RuntimeError('PRIVATE_TASK_ERROR')\n    return 'ok'\n")
        failure_report=root/'task-partial.json';failure_junit=root/'task-partial.xml'
        task_ci_fail=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--dataset-id',parallel_data['id'],'--dataset-version','1','--task-file',str(task_source),'--metric','exact_match','--report',str(failure_report),'--junit',str(failure_junit)],capture_output=True,text=True,timeout=10)
        assert task_ci_fail.returncode==2 and 'PRIVATE_TASK_ERROR' not in task_ci_fail.stdout+task_ci_fail.stderr
        partial_task=json.loads(failure_report.read_text())
        assert partial_task['completed_outputs']=={'0':'ok'} and partial_task['sample_id']=='1'
        partial_xml=ET.parse(failure_junit).getroot()
        assert partial_xml.attrib['tests']=='2' and partial_xml.attrib['errors']=='1' and partial_xml.attrib['skipped']=='1'
        assert [case.get('name') for case in partial_xml.findall('testcase')]==['0','execution.1']
        assert 'PRIVATE_TASK_ERROR' not in failure_junit.read_text()
        json_data=api('evaluation/datasets',dict(project_id='default',name='Structural JSON',base_version=0,samples=[dict(id='a',input='JSON_EQUAL_TEST',expected_output='{"a":1,"b":[true,null]}')]))
        json_run=wait_run(api('evaluation/experiments',dict(dataset_id=json_data['id'],dataset_version=1,settings=settings,prompt_template='{{input}}',metrics=['exact_match','json_equals','json_valid']))['id'])
        assert json_run['items'][0]['scores']==dict(exact_match=0,json_equals=1,json_valid=1)
        all_metrics=wait_run(api('evaluation/experiments',dict(dataset_id=json_data['id'],dataset_version=1,settings=settings,prompt_template='{{input}}',metrics=list(definitions)))['id'])
        assert all_metrics['status']=='completed' and set(all_metrics['items'][0]['scores'])==set(definitions)
        before_calls=len(requests)
        reject('evaluation/experiments',dict(dataset_id=dataset['id'],dataset_version=1,settings=settings,prompt_template='{{input}}',metrics=['json_equals']))
        assert len(requests)==before_calls
        duplicate_data=api('evaluation/datasets',dict(project_id='default',name='Ambiguous JSON',base_version=0,samples=[dict(id='a',input='JSON_DUPLICATE_TEST',expected_output='{"a":2}')]))
        duplicate_run=wait_run(api('evaluation/experiments',dict(dataset_id=duplicate_data['id'],dataset_version=1,settings=settings,prompt_template='{{input}}',metrics=['json_equals','json_valid']))['id'])
        assert duplicate_run['items'][0]['scores']==dict(json_equals=0,json_valid=1)
        bad_reference=api('evaluation/datasets',dict(project_id='default',name='Duplicate reference',base_version=0,samples=[dict(id='a',input='good',expected_output='{"a":1,"a":2}')]))
        before_calls=len(requests)
        reject('evaluation/experiments',dict(dataset_id=bad_reference['id'],dataset_version=1,settings=settings,prompt_template='{{input}}',metrics=['json_equals']))
        assert len(requests)==before_calls
        print('PASS JSON equality ignores formatting and rejects duplicate keys in output/reference')
        f1_data=api('evaluation/datasets',dict(project_id='default',name='Token overlap',base_version=0,samples=[dict(id='a',input='good',expected_output='answer: good extra')]))
        f1_run=wait_run(api('evaluation/experiments',dict(dataset_id=f1_data['id'],dataset_version=1,settings=settings,prompt_template='{{input}}',metrics=['whitespace_token_f1']))['id'])
        assert f1_run['status']=='completed' and f1_run['items'][0]['scores']['whitespace_token_f1']==0.8
        blank=api('evaluation/datasets',dict(project_id='default',name='Blank tokens',base_version=0,samples=[dict(id='a',input='good',expected_output=' \n')]))
        reject('evaluation/experiments',dict(dataset_id=blank['id'],dataset_version=1,settings=settings,prompt_template='{{input}}',metrics=['whitespace_token_f1']))
        f1_better=wait_run(api('evaluation/experiments',dict(dataset_id=f1_data['id'],dataset_version=1,settings=settings,prompt_template='{{input}} extra',metrics=['whitespace_token_f1']))['id'])
        f1_gain=api('evaluation/compare',dict(baseline_id=f1_run['id'],candidate_id=f1_better['id']))
        assert f1_gain['eligible'] and f1_gain['improvements']==1 and f1_gain['regressions']==0
        f1_decline=api('evaluation/compare',dict(baseline_id=f1_better['id'],candidate_id=f1_run['id']))
        assert not f1_decline['eligible'] and f1_decline['regressions']==1
        f1_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--dataset-id',f1_data['id'],'--dataset-version','1','--provider','local','--model','mock','--prompt-template','{{input}}','--metric','whitespace_token_f1']
        for minimum,expected_code in [('0.8',0),('0.9',1)]:
            checked=subprocess.run(f1_cli+['--min-score','whitespace_token_f1='+minimum],capture_output=True,text=True,timeout=10)
            result=json.loads(checked.stdout)
            assert checked.returncode==expected_code and result['passed']==(expected_code==0)
            assert result['thresholds']['whitespace_token_f1']['observed']==0.8
        before_calls=len(requests)
        offline=Studio(base).score_outputs(f1_data['id'],1,{'a':'answer: good'},['exact_match','whitespace_token_f1'])
        assert offline['mean_scores']==dict(exact_match=0,whitespace_token_f1=0.8)
        assert offline['dataset_sha256']==f1_data['sha256'] and offline['provider_calls']==0
        assert Studio(base).scored_outputs(offline['id'])==offline
        before_judge_plan_calls=len(requests)
        offline_judge_plan=Studio(base).plan_scored_output_judges(offline['id'],settings,'Evaluate correctness')
        assert len(requests)==before_judge_plan_calls and offline_judge_plan['provider_calls']==0
        assert offline_judge_plan['offline_score_source']['id']==offline['id']
        assert len(offline_judge_plan['offline_score_source']['snapshot_sha256'])==64
        assert offline_judge_plan['samples'][0]['output']=='answer: good'
        before_combined_judge_calls=len(requests)
        combined_evaluation=Studio(base).evaluate_scored_output_judges(offline['id'],settings,'Evaluate correctness',min_scores={'exact_match':1},min_judge_score=0,timeout=10)
        assert not combined_evaluation['passed'] and not combined_evaluation['deterministic']['passed']
        assert combined_evaluation['judge']['passed'] and combined_evaluation['judge']['run']['status']=='completed'
        assert combined_evaluation['judge_plan']['offline_score_source']['id']==offline['id']
        assert not combined_evaluation['automatic_promotion']
        assert len(requests)==before_combined_judge_calls+1
        reviewed=Studio(base).review_scored_output_judges(offline['id'],combined_evaluation['judge']['run']['id'],min_scores={'exact_match':0},min_judge_score=0)
        assert reviewed['passed'] and reviewed['provider_calls']==0
        assert not Studio(base).review_scored_output_judges(offline['id'],combined_evaluation['judge']['run']['id'],min_scores={'exact_match':1})['passed']
        assert len(requests)==before_combined_judge_calls+1
        review_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--review-score-id',offline['id'],'--review-judge-run-id',combined_evaluation['judge']['run']['id']]
        review_report=root/'review-report.json';review_junit=root/'review-report.xml'
        review_ci=subprocess.run(review_cli+['--min-score','exact_match=1','--report',str(review_report),'--junit',str(review_junit)],capture_output=True,text=True,timeout=15)
        assert review_ci.returncode==1,review_ci.stderr+review_ci.stdout
        review_receipt=json.loads(review_report.read_text())
        assert review_receipt['provider_calls']==0 and not review_receipt['passed'] and review_receipt['judge']['passed']
        review_xml=ET.parse(review_junit).getroot()
        assert review_xml.attrib['tests']=='2' and review_xml.attrib['failures']=='1' and review_xml.attrib['errors']=='0'
        passing_review=subprocess.run(review_cli+['--min-score','exact_match=0'],capture_output=True,text=True,timeout=15)
        assert passing_review.returncode==0,passing_review.stderr+passing_review.stdout
        invalid_review=subprocess.run(review_cli+['--provider','local'],capture_output=True,text=True,timeout=15)
        assert invalid_review.returncode==2
        assert len(requests)==before_combined_judge_calls+1
        before_calls+=1 # Explicit model judging is separate from zero-call offline scoring.
        combined_report=root/'combined-report.json';combined_junit=root/'combined-report.xml'
        combined_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--scored-judge-id',offline['id'],'--provider','local','--model','mock','--judge-preset','answer_relevance','--judge-preset-version','1','--judge-min-score','0','--min-score','exact_match=1','--report',str(combined_report),'--junit',str(combined_junit)]
        before_combined_cli=len(requests)
        combined_ci=subprocess.run(combined_cli,capture_output=True,text=True,timeout=15)
        assert combined_ci.returncode==1,combined_ci.stderr+combined_ci.stdout
        combined_receipt=json.loads(combined_report.read_text())
        assert not combined_receipt['deterministic']['passed'] and combined_receipt['judge']['passed']
        combined_xml=ET.parse(combined_junit).getroot()
        assert combined_xml.attrib['tests']=='2' and combined_xml.attrib['failures']=='1' and combined_xml.attrib['errors']=='0'
        assert len(requests)==before_combined_cli+1
        before_calls+=1
        slow_score=Studio(base).score_outputs(bigram_data['id'],1,{'a':'SLOW_JUDGE'},['character_bigram_f1'])
        partial_combined_report=root/'combined-partial.json';partial_combined_junit=root/'combined-partial.xml'
        slow_combined=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--scored-judge-id',slow_score['id'],'--provider','local','--model','mock','--judge-preset','answer_relevance','--judge-preset-version','1','--timeout','.2','--report',str(partial_combined_report),'--junit',str(partial_combined_junit)],capture_output=True,text=True,timeout=15)
        assert slow_combined.returncode==2,slow_combined.stderr+slow_combined.stdout
        partial_combined=json.loads(partial_combined_report.read_text())
        assert partial_combined['deterministic']['receipt']['id']==slow_score['id']
        assert partial_combined['judge_run_receipt']['id']==partial_combined['run_id']
        assert not partial_combined['judge_evidence_unavailable'] and partial_combined['judge_plan_id']
        assert ET.parse(partial_combined_junit).getroot().attrib['errors']=='1'
        before_calls+=1
        reject('evaluation/judge-plans',dict(dataset_id=f1_data['id'],dataset_version=1,settings=settings,rubric='Evaluate correctness',outputs={'a':'changed'},offline_score_id=offline['id']))
        score_catalog=Studio(base).list_scored_outputs('default',dataset_id=f1_data['id'],limit=1)
        assert score_catalog['provider_calls']==0 and not score_catalog['truncated']
        assert len(score_catalog['scores'])==1 and score_catalog['scores'][0]['id']==offline['id'], (score_catalog,offline['id'])
        assert score_catalog['scores'][0]['mean_scores']==offline['mean_scores']
        assert 'items' not in score_catalog['scores'][0] and 'output' not in json.dumps(score_catalog['scores'])
        assert not Studio(base).list_scored_outputs('foreign')['scores']
        assert not Studio(base).list_scored_outputs('default',dataset_id=f1_data['id'],offset=1)['scores']
        receipt_path=data/'evaluation'/'offline_scores'/f"{offline['id']}.json"
        original=receipt_path.read_bytes()
        tampered=json.loads(original);tampered['items'][0]['scores']['whitespace_token_f1']=1
        receipt_path.write_text(json.dumps(tampered))
        reject('evaluation/score/'+offline['id'])
        corrupted_catalog=Studio(base).list_scored_outputs('default',dataset_id=f1_data['id'])
        assert corrupted_catalog['invalid_receipts']>=1 and all(row['id']!=offline['id'] for row in corrupted_catalog['scores'])
        receipt_path.write_bytes(original)

        assert json.loads((data/'evaluation'/'offline_scores'/f"{offline['id']}.json").read_text())==offline
        for outputs in ({},{'foreign':'answer: good'},{'a':'x'*64001}):
            reject('evaluation/score',dict(dataset_id=f1_data['id'],dataset_version=1,metrics=['whitespace_token_f1'],outputs=outputs))
        reject('evaluation/score',dict(dataset_id=f1_data['id'],dataset_version=0,metrics=['whitespace_token_f1'],outputs={'a':'answer'}))
        assert len(requests)==before_calls
        offline_better=Studio(base).score_outputs(f1_data['id'],1,{'a':'answer: good extra'},['exact_match','whitespace_token_f1'])
        offline_gain=Studio(base).compare_scored_outputs(offline['id'],offline_better['id'])
        assert offline_gain['eligible'] and offline_gain['regressions']==0 and offline_gain['improvements']==2
        offline_decline=Studio(base).compare_scored_outputs(offline_better['id'],offline['id'])
        assert not offline_decline['eligible'] and offline_decline['regressions']==2
        offline_tie=Studio(base).compare_scored_outputs(offline['id'],offline['id'])
        assert not offline_tie['eligible'] and offline_tie['improvements']==0
        assert json.loads((data/'evaluation'/'offline_comparisons'/f"{offline_gain['id']}.json").read_text())==offline_gain
        foreign_metrics=Studio(base).score_outputs(f1_data['id'],1,{'a':'answer: good'},['exact_match'])
        reject('evaluation/score/compare',dict(baseline_id=offline['id'],candidate_id=foreign_metrics['id']))
        assert len(requests)==before_calls
        offline_gate=Studio(base).evaluate_outputs(f1_data['id'],1,{'a':'answer: good'},['exact_match','whitespace_token_f1'],min_scores={'whitespace_token_f1':0.8},baseline_id=offline['id'])
        assert offline_gate['passed'] and not offline_gate['comparison']['eligible']
        strict_gate=Studio(base).evaluate_outputs(f1_data['id'],1,{'a':'answer: good'},['exact_match','whitespace_token_f1'],baseline_id=offline['id'],require_improvement=True)
        assert not strict_gate['passed']
        threshold_gate=Studio(base).evaluate_outputs(f1_data['id'],1,{'a':'answer: good'},['whitespace_token_f1'],min_scores={'whitespace_token_f1':0.9})
        assert not threshold_gate['passed'] and threshold_gate['thresholds']['whitespace_token_f1']['observed']==0.8
        decline_gate=Studio(base).evaluate_outputs(f1_data['id'],1,{'a':'answer: good'},['exact_match','whitespace_token_f1'],baseline_id=offline_better['id'])
        assert not decline_gate['passed'] and decline_gate['comparison']['regressions']==2
        assert len(requests)==before_calls
        outputs_file=root/'ready-outputs.json';outputs_file.write_text(json.dumps({'a':'answer: good'}))
        offline_cli=[sys.executable,str(pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'),'--base-url',base,'--dataset-id',f1_data['id'],'--dataset-version','1','--outputs-file',str(outputs_file),'--metric','exact_match','--metric','whitespace_token_f1','--baseline-id',offline['id']]
        offline_report=root/'offline-report.json';offline_junit=root/'offline-failure.xml'
        passed_cli=subprocess.run(offline_cli+['--min-score','whitespace_token_f1=0.8','--report',str(offline_report)],capture_output=True,text=True,timeout=10)
        assert passed_cli.returncode==0 and json.loads(passed_cli.stdout)['passed']
        assert json.loads(offline_report.read_text())['receipt']['provider_calls']==0
        failed_cli=subprocess.run(offline_cli+['--min-score','whitespace_token_f1=0.9','--junit',str(offline_junit)],capture_output=True,text=True,timeout=10)
        assert failed_cli.returncode==1 and not json.loads(failed_cli.stdout)['passed']
        import xml.etree.ElementTree as ET
        assert ET.parse(offline_junit).getroot().get('failures')=='1'
        outputs_file.write_text('{"a":"first","a":"second"}')
        invalid_cli=subprocess.run(offline_cli,capture_output=True,text=True,timeout=10)
        assert invalid_cli.returncode==2 and json.loads(invalid_cli.stdout)['error']=='invalid_request_or_receipt'
        assert len(requests)==before_calls
        print('PASS offline SDK and CI gates, JSON/JUnit reports and duplicate sample rejection without provider calls')
        print('PASS token F1 partial overlap, paired improvement/regression and native CI thresholds')
        frozen_run=experiment(dataset);frozen=wait_run(frozen_run)
        assert frozen['dataset_version']==1 and frozen['dataset_sha256']==dataset['sha256'] and frozen['strict_quality'] and frozen['items'][0]['scores']['exact_match']==1
        exported=Studio(base).export_experiment(frozen['id'])
        assert exported['kind']=='experiment_export' and exported['schema_version']==1
        assert exported['dataset_sha256']==frozen['dataset_sha256'] and exported['mean_scores']==frozen['mean_scores']
        assert not exported['outputs_included'] and 'output' not in exported['items'][0]
        assert all(key not in exported for key in ('settings','prompt_template','prompt_snapshot'))
        answer_export=Studio(base).export_experiment(frozen['id'],include_outputs=True)
        assert answer_export['items'][0]['output']==frozen['items'][0]['output']
        import csv,io
        csv_rows=list(csv.DictReader(io.StringIO(Studio(base).export_experiment_csv(frozen['id']))))
        assert len(csv_rows)==len(frozen['items']) and csv_rows[0]['score_exact_match']=='1.0'
        assert 'output' not in csv_rows[0] and csv_rows[0]['dataset_sha256']==frozen['dataset_sha256']


        experiment_trace=trace_rows('experiment-'+frozen_run)[0];assert experiment_trace['id']==frozen['trace_id']
        assert [span['kind'] for span in experiment_trace['spans']]==['experiment','evaluation_item','model']
        assert experiment_trace['spans'][2]['parent_id']==experiment_trace['spans'][1]['id']
        assert_trace_tree(experiment_trace)
        assert dataset['samples'][0]['input'] not in json.dumps(experiment_trace)
        before_experiment_plan=len(requests)
        derived_plan=Studio(base).plan_experiment_judges(frozen_run,settings,'Check factual accuracy.')
        assert derived_plan['experiment_source']['id']==frozen_run and len(derived_plan['experiment_source']['snapshot_sha256'])==64
        assert derived_plan['samples'][0]['output']==frozen['items'][0]['output']
        assert Studio(base).judge_plan(derived_plan['id'])==derived_plan
        reject('evaluation/judge-plans',dict(experiment_id=frozen_run,dataset_id=dataset['id'],dataset_version=1,outputs={item['sample_id']:'Altered' for item in frozen['items']},settings=settings,rubric='Check factual accuracy.'))
        assert len(requests)==before_experiment_plan
        print('PASS experiment-derived frozen judge plan, source hash and rejection of substituted answers without provider calls')

        saved_prompt=api('evaluation/prompts',dict(project_id='default',name='Pinned task',base_version=0,template='{{input}}'))
        preview_calls=len(requests)
        prompt_preview=Studio(base).preview_prompt(saved_prompt['id'],1,'default','Literal {{contexts}}',contexts=['One','Two'])
        assert prompt_preview['prompt_sha256']==saved_prompt['sha256'] and prompt_preview['messages']==[dict(role='system',content=saved_prompt['system']),dict(role='user',content='Literal {{contexts}}')]
        assert len(requests)==preview_calls and prompt_preview['provider_calls']==0 and not prompt_preview['saved']
        reject('evaluation/prompts/'+saved_prompt['id']+'/versions/1/preview',dict(project_id='default',input='x'*65537))
        print('PASS pinned prompt preview renders provider messages without inference or saving a run',flush=True)
        playground_before=len(requests);playground_datasets=set((data/'evaluation'/'datasets').iterdir());playground_runs=set((data/'evaluation'/'runs').iterdir())
        reject('evaluation/playground',dict(settings=settings,prompt_ref=dict(id=saved_prompt['id'],version=1),prompt_sha256='0'*64,input='Question'))
        reject('evaluation/playground',dict(settings=dict(settings,allow_writes=True),prompt_ref=dict(id=saved_prompt['id'],version=1),prompt_sha256=saved_prompt['sha256'],input='Question'))
        assert set((data/'evaluation'/'datasets').iterdir())==playground_datasets and set((data/'evaluation'/'runs').iterdir())==playground_runs and len(requests)==playground_before
        playground=wait_run(Studio(base).start_playground(saved_prompt['id'],1,saved_prompt['sha256'],'Playground question',settings)['id'])
        assert playground['status']=='completed' and playground['playground'] and not playground['strict_quality'] and playground['metrics']==[] and playground['mean_scores']=={}
        assert playground['prompt_snapshot']['sha256']==saved_prompt['sha256'] and playground['items'][0]['output']=='answer: Playground question' and len(requests)==playground_before+1
        playground_sample=api('evaluation/datasets/'+playground['dataset_id']+'/versions/1');assert playground_sample['sha256']==playground['dataset_sha256'] and playground_sample['samples'][0]['metadata']['kind']=='playground'
        assert api('evaluation/experiments/'+playground['id']+'/export')['playground']
        reject('evaluation/compare',dict(baseline_id=playground['id'],candidate_id=playground['id']))
        playground_guard_before=len(requests)
        blocked_playground=wait_run(Studio(base).start_playground(saved_prompt['id'],1,saved_prompt['sha256'],'private-marker',dict(settings,guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=policy_hash,action='block')))['id'])
        assert blocked_playground['status']=='failed' and len(requests)==playground_guard_before and not blocked_playground['strict_quality']
        print('PASS single-input Playground saves source pins/output/trace, refuses unreviewed hash and writes before persistence, blocks guarded input and does not certify unscored answers',flush=True)
        chat_messages=[dict(role='user',content='Example question'),dict(role='assistant',content='Example answer'),dict(role='user',content='Q {{input}} / {{contexts}}')]
        chat_prompt=Studio(base).save_chat_prompt('default','Chat template',chat_messages,system='Use examples.')
        chat_preview=Studio(base).preview_prompt(chat_prompt['id'],1,'default','Question',contexts=['Context'])
        assert chat_preview['messages']==[dict(role='system',content='Use examples.'),*chat_messages[:2],dict(role='user',content='Q Question / Context')]
        chat_before=len(requests);chat_run=wait_run(Studio(base).start_playground(chat_prompt['id'],1,chat_prompt['sha256'],'Question',settings,contexts=['Context'])['id'])
        assert chat_run['status']=='completed' and chat_run['items'][0]['output']=='answer: Q Question / Context' and len(requests)==chat_before+1
        assert requests[-1]['messages']==chat_preview['messages'] and chat_run['prompt_snapshot']['messages']==chat_messages
        chat_dataset=api('evaluation/datasets',dict(project_id='default',name='Chat template test',base_version=0,samples=[dict(id='one',input='Question',contexts=['Context'],expected_output='answer: Q Question / Context')]))
        chat_eval=wait_run(api('evaluation/experiments',dict(dataset_id=chat_dataset['id'],dataset_version=1,settings=settings,prompt_ref=dict(id=chat_prompt['id'],version=1),metrics=['exact_match']))['id'])
        assert chat_eval['strict_quality'] and chat_eval['mean_scores']['exact_match']==1 and requests[-1]['messages']==chat_preview['messages']
        chat_block_before=len(requests);chat_blocked=wait_run(Studio(base).start_playground(chat_prompt['id'],1,chat_prompt['sha256'],'private-marker',dict(settings,guardrails=dict(input_policy_sha256=policy_hash,output_policy_sha256=policy_hash,action='block')))['id'])
        assert chat_blocked['status']=='failed' and chat_blocked['items'][0]['error']=='guardrail_blocked' and len(requests)==chat_block_before
        reject('evaluation/prompts',dict(project_id='default',name='Invalid roles',base_version=0,template='',messages=[dict(role='assistant',content='{{input}}')]))
        print('PASS chat-message prompt pins/preview/provider roles, Playground and scored dataset execution, final-message guardrail enforcement and invalid role rejection',flush=True)
        catalog_calls=len(requests);legacy_runs=api('evaluation/experiments?project_id=default')['runs'];first_runs=Studio(base).experiments('default',limit=1);second_runs=Studio(base).experiments('default',offset=1,limit=1)
        assert first_runs['total']==len(legacy_runs) and first_runs['has_more'] and first_runs['runs'][0]['id']!=second_runs['runs'][0]['id']
        filtered_runs=Studio(base).experiments('default',status='completed',provider='local',model='MOCK',dataset_id=chat_dataset['id'],playground=False)
        assert filtered_runs['total']==1 and filtered_runs['runs'][0]['id']==chat_eval['id']
        playground_runs=Studio(base).experiments('default',playground=True);assert playground_runs['total']==4 and all(row['playground'] for row in playground_runs['runs'])
        reject('evaluation/experiments?project_id=default&limit=101');reject('evaluation/experiments?project_id=default&status=unknown')
        assert len(requests)==catalog_calls
        print('PASS paginated experiment catalog filters status/model/provider/dataset/Playground before totals, preserves legacy list and never invokes provider',flush=True)
        scored_playground_calls=len(requests)
        scored_playground=wait_run(Studio(base).start_playground(chat_prompt['id'],1,chat_prompt['sha256'],'Question',settings,contexts=['Context'],expected_output='answer: Q Question / Context',metrics=['exact_match','whitespace_token_f1'])['id'])
        assert scored_playground['playground'] and scored_playground['strict_quality'] and scored_playground['mean_scores']==dict(exact_match=1.0,whitespace_token_f1=1.0) and len(requests)==scored_playground_calls+1
        scored_sample=api('evaluation/datasets/'+scored_playground['dataset_id']+'/versions/1')['samples'][0];assert scored_sample['expected_output']=='answer: Q Question / Context'
        scored_artifacts=set((data/'evaluation'/'datasets').iterdir());scored_calls=len(requests)
        reject('evaluation/playground',dict(settings=settings,prompt_ref=dict(id=chat_prompt['id'],version=1),prompt_sha256=chat_prompt['sha256'],input='Question',metrics=['exact_match']))
        assert len(requests)==scored_calls and set((data/'evaluation'/'datasets').iterdir())==scored_artifacts
        print('PASS scored Playground pins reference/metrics, persists verified exact/F1 scores and rejects missing reference before saving or inference',flush=True)
        escaped_calls=len(requests)
        escaped_run=wait_run(Studio(base).start_playground(chat_prompt['id'],1,chat_prompt['sha256'],'ESCAPED_OUTPUT_FIXTURE',settings,metrics=['json_valid'])['id'])
        escaped_item=escaped_run['items'][0]
        assert escaped_run['status']=='failed' and not escaped_run['strict_quality'] and not escaped_run['mean_scores'] and not escaped_item['scores']
        assert escaped_item['output_truncated'] and escaped_item['error']=='incomplete_or_oversize_output' and 0<len(escaped_item['output'])<65536
        assert len(json.dumps(escaped_item['output']).encode())<=128*1024+16 and len(requests)==escaped_calls+1
        persisted_escaped=json.loads((data/'evaluation'/'runs'/(escaped_run['id']+'.json')).read_text())
        assert persisted_escaped['items'][0]['output']==escaped_item['output'] and persisted_escaped['status']=='failed'
        escaped_export=Studio(base).export_experiment(escaped_run['id'])
        assert not escaped_export['outputs_included'] and 'output' not in escaped_export['items'][0] and escaped_export['items'][0]['output_truncated']
        assert escaped_export['provider']==settings['provider'] and escaped_export['model']==settings['model'] and escaped_export['trace_id']==escaped_run['trace_id']
        assert escaped_export['prompt_ref']==dict(id=chat_prompt['id'],version=1,sha256=chat_prompt['sha256'])
        escaped_csv=list(csv.DictReader(io.StringIO(Studio(base).export_experiment_csv(escaped_run['id']))))
        assert len(escaped_csv)==1 and escaped_csv[0]['model']==settings['model'] and escaped_csv[0]['provider']==settings['provider']
        assert escaped_csv[0]['trace_id']==escaped_run['trace_id'] and escaped_csv[0]['prompt_id']==chat_prompt['id'] and escaped_csv[0]['prompt_version']=='1' and escaped_csv[0]['prompt_sha256']==chat_prompt['sha256']
        assert 'output' not in escaped_csv[0] and not escaped_csv[0]['score_json_valid']
        assert all(key not in escaped_export for key in ('settings','prompt_template','prompt_snapshot'))
        escaped_output_export=Studio(base).export_experiment(escaped_run['id'],include_outputs=True)
        assert escaped_output_export['items'][0]['output']==escaped_item['output'] and not escaped_output_export['strict_quality']
        reject('evaluation/compare',dict(baseline_id=escaped_run['id'],candidate_id=escaped_run['id']))
        assert len(requests)==escaped_calls+1
        print('PASS escaped output is durably bounded and cannot yield scores or strict quality',flush=True)
        capacity_root=data/'evaluation'/'runs';capacity_source=json.loads((capacity_root/(scored_playground['id']+'.json')).read_text());capacity_files=[];capacity_calls=len(requests);capacity_datasets=set((data/'evaluation'/'datasets').iterdir())
        try:
            for capacity_index in range(1000-len(list(capacity_root.glob('*.json')))):
                capacity_id='cafe-'+format(capacity_index,'x');capacity_path=capacity_root/(capacity_id+'.json');capacity_path.write_text(json.dumps(dict(capacity_source,id=capacity_id)));capacity_files.append(capacity_path)
            reject('evaluation/playground',dict(settings=settings,prompt_ref=dict(id=chat_prompt['id'],version=1),prompt_sha256=chat_prompt['sha256'],input='No space'))
            assert len(list(capacity_root.glob('*.json')))==1000 and set((data/'evaluation'/'datasets').iterdir())==capacity_datasets and len(requests)==capacity_calls
            assert Studio(base).experiments('default',limit=1)['has_more']
        finally:
            for capacity_path in capacity_files:capacity_path.unlink()
        print('PASS experiment capacity rejects full catalog before saving Playground sample or calling provider and preserves readable history',flush=True)






        pinned_id=api('evaluation/experiments',dict(dataset_id=dataset['id'],dataset_version=1,settings=settings,prompt_ref=dict(id=saved_prompt['id'],version=1),metrics=['exact_match']))['id']
        next_prompt=api('evaluation/prompts',dict(id=saved_prompt['id'],project_id='default',name='Pinned task',base_version=1,template='new {{input}}',system='Custom system instruction'))
        pinned=wait_run(pinned_id)
        assert pinned['prompt_snapshot']==saved_prompt and pinned['prompt_template']=='{{input}}'
        assert pinned['items'][0]['output']==frozen['items'][0]['output']
        assert api(f"evaluation/prompts/{saved_prompt['id']}/versions/1")==saved_prompt
        assert next(row for row in api('evaluation/prompts?project_id=default')['prompts'] if row['id']==saved_prompt['id'])['version']==2
        reject('evaluation/prompts',dict(id=saved_prompt['id'],project_id='default',name='Stale',base_version=1,template='{{input}}'),status=409)
        reject('evaluation/experiments',dict(dataset_id=dataset['id'],dataset_version=1,settings=settings,prompt_ref=dict(id=saved_prompt['id'],version=1),prompt_template='{{input}}',metrics=['exact_match']))
        custom=wait_run(api('evaluation/experiments',dict(dataset_id=dataset['id'],dataset_version=1,settings=settings,prompt_ref=dict(id=saved_prompt['id'],version=2),metrics=['exact_match']))['id'])
        assert custom['prompt_snapshot']==next_prompt
        assert any(r['messages'][0]['content']=='Custom system instruction' for r in requests)
        prompt_file=data/'evaluation'/'prompts'/saved_prompt['id']/'00000000000000000002.json'
        corrupt=dict(next_prompt,system='corrupt');prompt_file.write_text(json.dumps(corrupt))
        reject(f"evaluation/prompts/{saved_prompt['id']}/versions/2")
        prompt_file.write_text(json.dumps(next_prompt))
        history=api(f"evaluation/prompts/{saved_prompt['id']}/versions?limit=1")
        assert history['total']==2 and history['versions'][0]['version']==2
        assert api(f"evaluation/prompts/{saved_prompt['id']}/versions?offset=1&limit=1")['versions'][0]['sha256']==saved_prompt['sha256']
        reject(f"evaluation/prompts/{saved_prompt['id']}/versions?limit=101")
        origin=dict(id=saved_prompt['id'],version=1,sha256=saved_prompt['sha256'])
        variant=api('evaluation/prompts',dict(project_id='default',name='Variant',base_version=0,template='variant {{input}}',origin=origin))
        assert variant['origin']==origin
        variant_revision=api('evaluation/prompts',dict(id=variant['id'],project_id='default',name='Variant',base_version=1,template='revised {{input}}'))
        assert variant_revision['origin']==origin
        reject('evaluation/prompts',dict(id=variant['id'],project_id='default',name='Variant',base_version=2,template='{{input}}',origin=dict(origin,version=2)))
        reject('evaluation/prompts',dict(project_id='default',name='Bad origin',base_version=0,template='{{input}}',origin=dict(origin,sha256='0'*64)))
        variant_run=wait_run(api('evaluation/experiments',dict(dataset_id=dataset['id'],dataset_version=1,settings=settings,prompt_ref=dict(id=variant['id'],version=2),metrics=['exact_match']))['id'])
        assert variant_run['prompt_snapshot']['origin']==origin
        print('PASS pinned prompt variant ancestry survives updates and reaches experiment receipts')
        print('PASS immutable prompt versions, pinned experiment receipts, custom system instructions, stale writes and tamper rejection')
        pair_dataset=api('evaluation/datasets',dict(project_id='default',name='Pairs',base_version=0,samples=[dict(id=k,input='PAIR_TEST '+k,expected_output='ok') for k in ['a','b','c']]))
        baseline=wait_run(experiment(pair_dataset,'baseline'));candidate=wait_run(experiment(pair_dataset,'candidate'));better=wait_run(experiment(pair_dataset,'better'))
        assert candidate['mean_scores']['exact_match']>baseline['mean_scores']['exact_match']
        rejected=api('evaluation/compare',dict(baseline_id=baseline['id'],candidate_id=candidate['id']))
        assert not rejected['eligible'] and rejected['regressions']==1 and rejected['improvements']==2
        accepted=api('evaluation/compare',dict(baseline_id=baseline['id'],candidate_id=better['id']))
        assert accepted['eligible'] and accepted['regressions']==0
        direct_matrix_candidates=[dict(label=model,sha256=pair_dataset['sha256'],request=dict(dataset_id=pair_dataset['id'],dataset_version=1,settings=dict(settings,model=model),metrics=['exact_match'],prompt_template='{{input}}')) for model in ['baseline','candidate','better']]
        direct_matrix_process=subprocess.run(['node','scripts/test-studio-matrix-native.js',base,json.dumps(direct_matrix_candidates)],capture_output=True,text=True,timeout=30)
        assert direct_matrix_process.returncode==0,direct_matrix_process.stderr
        direct_matrix_receipt=json.loads(direct_matrix_process.stdout)
        assert not direct_matrix_receipt['passed'] and Studio(base).experiment_matrix(direct_matrix_receipt['id'])['variants'][1]['comparison']['regressions']==1

        def wait_matrix_job(job_id):
            until=time.time()+10
            while time.time()<until:
                job=Studio(base).matrix_job(job_id)
                if job['status']!='running':return job
                time.sleep(.03)
            raise AssertionError(job)
        job_variants=[dict(label=row['label'],sha256=row['sha256'],request=row['request']) for row in direct_matrix_candidates]
        before_job=len(requests)
        invalid_job=json.loads(json.dumps(job_variants));invalid_job[-1]['request']['prompt_template']='no input placeholder'
        try:Studio(base).start_matrix_job('default',invalid_job);raise AssertionError('late invalid variant admitted')
        except EvaluationError as error:assert error.reason=='http_400'
        assert len(requests)==before_job
        invalid_prompt_job=json.loads(json.dumps(job_variants));invalid_prompt_job[-1]['request'].pop('prompt_template');invalid_prompt_job[-1]['request']['prompt_ref']=dict(id=saved_prompt['id'],version=1);invalid_prompt_job[-1]['prompt_sha256']='0'*64
        try:Studio(base).start_matrix_job('default',invalid_prompt_job);raise AssertionError('changed pinned prompt admitted')
        except EvaluationError as error:assert error.reason=='http_400'
        assert len(requests)==before_job
        server_job=Studio(base).start_matrix_job('default',job_variants)
        completed_job=wait_matrix_job(server_job['id'])
        assert completed_job['status']=='completed' and len(requests)==before_job+9,completed_job
        assert not completed_job['automatic_replay'] and not completed_job['automatic_promotion']
        server_matrix=Studio(base).experiment_matrix(completed_job['matrix_id'])
        assert not server_matrix['passed'] and server_matrix['variants'][1]['comparison']['regressions']==1
        assert completed_job['id'] in [row['id'] for row in Studio(base).matrix_jobs('default')['jobs']]

        recovery_variants=[dict(label=label,sha256=dataset['sha256'],request=dict(dataset_id=dataset['id'],dataset_version=dataset['version'],settings=settings,metrics=['exact_match'],prompt_template=template)) for label,template in [('slow','SLOW_SAVE {{input}}'),('next','{{input}}')]]
        save_job=Studio(base).start_matrix_job('default',recovery_variants)
        save_job_path=data/'evaluation'/'matrix-jobs'/(save_job['id']+'.json')
        save_job_private=json.loads(save_job_path.read_text())
        reserved_matrix_path=data/'evaluation'/'matrices'/(save_job_private['matrix_id']+'.json');reserved_matrix_path.mkdir()
        failed_save=wait_matrix_job(save_job['id']);assert failed_save['status']=='failed' and all(row['run_id'] for row in failed_save['variants']),failed_save
        before_resume=len(requests);reserved_matrix_path.rmdir()
        saved_job_bytes=save_job_path.read_bytes();changed_route=json.loads(saved_job_bytes);changed_route['provider_routes'][0]='0'*64;save_job_path.write_text(json.dumps(changed_route))
        try:
            try:Studio(base).resume_matrix_job(save_job['id']);raise AssertionError('changed provider route resumed')
            except EvaluationError as error:assert error.reason=='http_400'
            assert len(requests)==before_resume
        finally:save_job_path.write_bytes(saved_job_bytes)
        completed_owned_path=data/'evaluation'/'runs'/(failed_save['variants'][0]['run_id']+'.json');owned_bytes=completed_owned_path.read_bytes();completed_owned_path.unlink()
        try:
            try:Studio(base).resume_matrix_job(save_job['id']);raise AssertionError('missing completed receipt replayed')
            except EvaluationError as error:assert error.reason=='http_400'
            assert len(requests)==before_resume
        finally:completed_owned_path.write_bytes(owned_bytes)
        Studio(base).resume_matrix_job(save_job['id']);resumed_save=wait_matrix_job(save_job['id'])
        assert resumed_save['status']=='completed' and len(requests)==before_resume,resumed_save
        assert resumed_save['variants']==failed_save['variants']
        assert Studio(base).experiment_matrix(resumed_save['matrix_id'])['id']==save_job_private['matrix_id']
        print('PASS server-owned matrix sequencing, all-variant preflight, persisted IDs, conservative gate and save-only resume without inference replay',flush=True)

        quota_root=data/'evaluation'/'matrix-jobs'
        quota_original=json.loads((quota_root/(completed_job['id']+'.json')).read_text())
        quota_files=[];before_quota=len(requests)
        try:
            for quota_index in range(1000-len(list(quota_root.glob('*.json')))):
                quota_id='f00d-'+format(quota_index,'x');quota_file=quota_root/(quota_id+'.json')
                quota_fixture=dict(quota_original,id=quota_id);quota_file.write_text(json.dumps(quota_fixture));quota_files.append(quota_file)
            assert len(list(quota_root.glob('*.json')))==1000
            try:Studio(base).start_matrix_job('default',job_variants);raise AssertionError('full matrix catalog admitted job')
            except EvaluationError as error:assert error.reason=='http_400'
            assert len(list(quota_root.glob('*.json')))==1000 and len(requests)==before_quota
            assert Studio(base).matrix_jobs('default',limit=1)['has_more']
            assert Studio(base).matrix_job(completed_job['id'])==completed_job
        finally:
            for quota_file in quota_files:quota_file.unlink()
        print('PASS full matrix job catalog rejects new admission before inference and remains readable',flush=True)

        saved_experiment_comparison=accepted
        saved_native_matrix=Studio(base).save_experiment_matrix('default',baseline['id'],[dict(label='baseline',run_id=baseline['id']),dict(label='regression',run_id=candidate['id']),dict(label='gain',run_id=better['id'])])
        assert saved_native_matrix['kind']=='experiment_matrix' and not saved_native_matrix['passed'] and not saved_native_matrix['automatic_promotion']
        assert saved_native_matrix['variants'][1]['comparison']['regressions']==1
        assert saved_native_matrix['variants'][2]['comparison']['eligible']
        assert Studio(base).experiment_matrix(saved_native_matrix['id'])==saved_native_matrix
        assert Studio(base).matrix_job(completed_job['id'])==completed_job
        positive_matrix=Studio(base).save_experiment_matrix('default',baseline['id'],[dict(label='baseline',run_id=baseline['id']),dict(label='gain',run_id=better['id'])]);assert positive_matrix['passed']
        matrix_catalog=Studio(base).experiment_matrices('default',limit=1)
        next_matrix_page=Studio(base).experiment_matrices('default',offset=1,limit=1)
        assert matrix_catalog['has_more'] and not matrix_catalog['summaries_verified']
        assert matrix_catalog['matrices'][0]['id']!=next_matrix_page['matrices'][0]['id']
        assert 'passed' not in matrix_catalog['matrices'][0] and 'variants' not in matrix_catalog['matrices'][0]

        matrix_path=data/'evaluation'/'matrices'/(saved_native_matrix['id']+'.json');matrix_original=matrix_path.read_bytes()
        altered_matrix=json.loads(matrix_original);altered_matrix['passed']=True;matrix_path.write_text(json.dumps(altered_matrix))
        try:Studio(base).experiment_matrix(saved_native_matrix['id']);raise AssertionError('forged matrix quality accepted')
        except EvaluationError as error:assert error.reason=='http_400'
        finally:matrix_path.write_bytes(matrix_original)

        comparison_catalog=Studio(base).comparisons('default',limit=100)
        assert not comparison_catalog['summaries_verified']
        assert accepted['id'] in [row['id'] for row in comparison_catalog['comparisons']]
        first_page=Studio(base).comparisons('default',limit=1)
        second_page=Studio(base).comparisons('default',offset=1,limit=1)
        assert first_page['has_more'] and first_page['comparisons'][0]['id']!=second_page['comparisons'][0]['id']
        assert all('pairs' not in row and 'eligible' not in row for row in comparison_catalog['comparisons'])

        assert Studio(base).comparison(accepted['id'])==accepted
        assert Studio(base).comparison(rejected['id'])==rejected
        comparison_csv=list(csv.DictReader(io.StringIO(Studio(base).export_comparison_csv(rejected['id']))))
        assert len(comparison_csv)==len(rejected['pairs'])
        assert sum(row['change']=='regression' for row in comparison_csv)==1
        assert all(row['eligible']=='False' and row['reason']=='paired_regression' for row in comparison_csv)

        comparison_path=data/'evaluation'/'comparisons'/(accepted['id']+'.json')
        comparison_original=comparison_path.read_bytes()
        altered=json.loads(comparison_original);altered['pairs'][0]['candidate']=0.25
        comparison_path.write_text(json.dumps(altered))
        try:Studio(base).comparison(accepted['id']);raise AssertionError('tampered saved comparison accepted')
        except EvaluationError as error:assert error.reason=='http_400'
        finally:comparison_path.write_bytes(comparison_original)

        matrix_requests=[dict(dataset_id=pair_dataset['id'],dataset_version=1,
            settings=dict(settings,model=model),prompt_template='{{input}}',metrics=['exact_match'])
            for model in ['baseline','candidate','better']]
        matrix=Studio(base).evaluate_matrix(matrix_requests,persist=True,labels=['baseline','regression','gain'],
                                            timeout=10,poll_interval=0.01,require_improvement=True)
        assert not matrix['automatic_promotion']
        assert not matrix['native_matrix']['passed']
        assert Studio(base).experiment_matrix(matrix['native_matrix']['id'])==matrix['native_matrix']
        assert matrix['variants'][1]['result']['comparison']['regressions']==1
        assert not matrix['variants'][1]['result']['passed']
        assert matrix['variants'][2]['result']['passed']
        assert len({row['result']['run']['id'] for row in matrix['variants']})==3
        for row in matrix['variants']:
            assert Studio(base).run(row['result']['run']['id'])['dataset_sha256']==pair_dataset['sha256']
        print('PASS SDK model matrix persists distinct native runs and rejects paired regression despite improved mean')
        ci_script=pathlib.Path(__file__).resolve().parent/'evaluate-studio.py'
        matrix_file=root/'matrix.json';matrix_report=root/'matrix-report.json';matrix_junit=root/'matrix.xml'
        matrix_file.write_text(json.dumps(dict(requests=matrix_requests,labels=['baseline','regression','gain'])))
        matrix_command=[os.sys.executable,str(ci_script),'--base-url',base,'--matrix-file',str(matrix_file),'--require-improvement']
        matrix_ci=subprocess.run(matrix_command+['--persist-matrix','--report',str(matrix_report),'--junit',str(matrix_junit)],capture_output=True,text=True,timeout=15)
        assert matrix_ci.returncode==1 and not json.loads(matrix_ci.stdout)['passed']
        assert not Studio(base).experiment_matrix(json.loads(matrix_ci.stdout)['native_matrix_id'])['passed']
        matrix_receipt=json.loads(matrix_report.read_text())
        assert len(matrix_receipt['variants'])==3 and matrix_receipt['variants'][2]['result']['passed']
        import xml.etree.ElementTree as ET
        matrix_xml=ET.parse(matrix_junit).getroot()
        assert matrix_xml.get('tests')=='3' and matrix_xml.get('failures')=='1' and matrix_xml.get('errors')=='0'
        matrix_file.write_text(json.dumps(dict(requests=[matrix_requests[0],matrix_requests[2]])))
        matrix_good=subprocess.run(matrix_command,capture_output=True,text=True,timeout=15)
        assert matrix_good.returncode==0 and json.loads(matrix_good.stdout)['passed']
        bad_matrix=dict(matrix_requests[2],dataset_version=2)
        matrix_file.write_text(json.dumps(dict(requests=[matrix_requests[0],bad_matrix])))
        calls_before=len(requests)
        bad_report=root/'bad-matrix-report.json'
        matrix_bad=subprocess.run(matrix_command+['--report',str(bad_report)],capture_output=True,text=True,timeout=10)
        assert matrix_bad.returncode==2 and len(requests)==calls_before
        assert json.loads(bad_report.read_text())['variants']==[]
        matrix_file.write_text('{"requests":[],"requests":[]}')
        duplicate_matrix=subprocess.run(matrix_command,capture_output=True,text=True,timeout=10)
        assert duplicate_matrix.returncode==2 and len(requests)==calls_before
        unavailable=dict(matrix_requests[2],settings=dict(settings,provider='missing-matrix-provider',model='better'))
        matrix_file.write_text(json.dumps(dict(requests=[matrix_requests[0],unavailable])))
        partial_report=root/'partial-matrix-report.json';partial_junit=root/'partial-matrix.xml'
        partial_matrix=subprocess.run(matrix_command+['--report',str(partial_report),'--junit',str(partial_junit)],capture_output=True,text=True,timeout=15)
        assert partial_matrix.returncode==2
        partial_receipt=json.loads(partial_report.read_text())
        assert partial_receipt['variant_index']==1 and len(partial_receipt['variants'])==1
        assert Studio(base).run(partial_receipt['variants'][0]['result']['run']['id'])['status']=='completed'
        partial_xml=ET.parse(partial_junit).getroot()
        assert partial_xml.get('tests')=='2' and partial_xml.get('errors')=='1'
        print('PASS matrix CI success/regression/error exits, per-variant JUnit, full JSON and malformed preflight without provider calls')
        def ci(model, baseline_id, *extra):
            command=[os.sys.executable,str(ci_script),'--base-url',base,'--dataset-id',pair_dataset['id'],'--dataset-version','1','--provider','local','--model',model,'--metric','exact_match']
            if '--prompt-id' not in extra:command+=['--prompt-template','{{input}}']
            if baseline_id:command+=['--baseline-id',baseline_id]
            result=subprocess.run(command+list(extra),capture_output=True,text=True,timeout=10)
            return result.returncode,json.loads(result.stdout)
        report=root/'ci-evaluation.json';junit=root/'ci-evaluation.xml'
        ci_code,ci_good=ci('better',baseline['id'],'--report',str(report),'--junit',str(junit))
        assert ci_code==0 and ci_good['passed'] and ci_good['comparison']['eligible']
        ci_artifact=json.loads(report.read_text());assert ci_artifact['run']['dataset_sha256']==pair_dataset['sha256']
        assert ci_artifact['comparison']['candidate_id']==ci_good['run_id']
        import xml.etree.ElementTree as ET
        assert ET.parse(junit).getroot().get('failures')=='0'
        bad_junit=root/'ci-regression.xml'
        ci_code,ci_bad=ci('candidate',baseline['id'],'--junit',str(bad_junit))
        assert ci_code==1 and not ci_bad['passed'] and ci_bad['comparison']['regressions']==1
        assert ET.parse(bad_junit).getroot().get('failures')=='1'
        assert ET.parse(bad_junit).find('.//failure') is not None
        ci_code,ci_tie=ci('better',better['id'])
        assert ci_code==0 and ci_tie['comparison']['reason']=='no_improvement' and not ci_tie['comparison']['eligible']
        ci_code,ci_strict_tie=ci('better',better['id'],'--require-improvement')
        assert ci_code==1 and ci_strict_tie['gate_policy']=='strict_improvement'
        ci_code,ci_minimum=ci('candidate',None,'--min-score','exact_match=1')
        assert ci_code==1 and not ci_minimum['thresholds']['exact_match']['passed']
        ci_code,ci_minimum=ci('better',None,'--min-score','exact_match=1')
        assert ci_code==0 and ci_minimum['thresholds']['exact_match']['passed']
        calls_before=len(requests);ci_code,ci_existing=ci('better',baseline['id'],'--report',str(report))
        assert ci_code==2 and ci_existing['error']=='artifact_path_exists_or_duplicate' and len(requests)==calls_before
        assert json.loads(report.read_text())==ci_artifact
        pinned_report=root/'ci-pinned-prompt.json'
        ci_code,ci_pinned=ci('better',None,'--prompt-id',saved_prompt['id'],'--prompt-version','1','--min-score','exact_match=1','--report',str(pinned_report))
        assert ci_code==0 and json.loads(pinned_report.read_text())['run']['prompt_snapshot']==saved_prompt
        ci_code,ci_timeout=ci('better',None,'--timeout','0.000000000001')
        assert ci_code==2 and ci_timeout['run_id'] and ci_timeout['error']=='evaluation_timeout'
        assert wait_run(ci_timeout['run_id'])['status']=='cancelled'
        print('PASS native evaluation CI exit codes, conservative regression/tie gate, retained JSON/JUnit reports, overwrite prevention and timeout cancellation')
        tied=api('evaluation/compare',dict(baseline_id=better['id'],candidate_id=better['id']))
        assert not tied['eligible'] and tied['reason']=='no_improvement'
        mismatch=wait_run(experiment(revision))
        try:api('evaluation/compare',dict(baseline_id=frozen_run,candidate_id=mismatch['id']));raise AssertionError('different snapshots compared')
        except urllib.error.HTTPError as e:assert e.code==400
        slow_data=api('evaluation/datasets',dict(project_id='default',name='Slow',base_version=0,samples=[dict(id='slow',input='SLOW',expected_output='never')]))
        slow_run=experiment(slow_data);api('evaluation/experiments/'+slow_run+'/cancel',{})
        cancelled=wait_run(slow_run);assert cancelled['status']=='cancelled' and not cancelled['strict_quality']
        cancelled_trace=trace_rows('experiment-'+slow_run)[0];assert cancelled_trace['status']=='cancelled';assert_trace_tree(cancelled_trace)
        unrelated_run=experiment(slow_data)
        cancel_variants=[dict(label=label,sha256=slow_data['sha256'],request=dict(dataset_id=slow_data['id'],dataset_version=1,settings=settings,metrics=['exact_match'],prompt_template='{{input}}')) for label in ['first','never admitted']]
        cancel_job=Studio(base).start_matrix_job('default',cancel_variants)
        until=time.time()+5
        while time.time()<until:
            cancel_progress=Studio(base).matrix_job(cancel_job['id'])
            if cancel_progress['variants'][0]['run_id']:break
            time.sleep(.01)
        assert Studio(base).cancel_matrix_job(cancel_job['id'])['cancel_requested']
        cancelled_matrix=wait_matrix_job(cancel_job['id'])
        assert cancelled_matrix['status']=='cancelled' and cancelled_matrix['variants'][1]['run_id'] is None,cancelled_matrix
        assert api('evaluation/experiments/'+cancelled_matrix['variants'][0]['run_id'])['status']=='cancelled'
        assert wait_run(unrelated_run)['status']=='completed'
        print('PASS matrix cancellation owns only its active experiment and never admits later variants',flush=True)
        broken_data=api('evaluation/datasets',dict(project_id='default',name='Broken',base_version=0,samples=[dict(id='broken',input='TOKEN_LIMIT_TEXT',expected_output='partial text')]))
        failed_run=wait_run(experiment(broken_data));assert failed_run['status']=='failed' and failed_run['items'][0]['scores']=={} and not failed_run['strict_quality'] and failed_run['items'][0]['output']=='partial text' and failed_run['items'][0]['output_truncated']
        failed_export=Studio(base).export_experiment(failed_run['id'])
        assert failed_export['status']=='failed' and not failed_export['strict_quality']
        assert failed_export['items'][0]['has_error'] and failed_export['items'][0]['output_truncated']
        assert failed_export['items'][0]['scores']=={} and 'error' not in failed_export['items'][0] and 'usage' not in failed_export['items'][0]
        failed_csv=list(csv.DictReader(io.StringIO(Studio(base).export_experiment_csv(failed_run['id'],include_outputs=True))))
        assert failed_csv[0]['run_status']=='failed' and failed_csv[0]['has_error']=='True'
        assert failed_csv[0]['output']=='partial text' and failed_csv[0]['score_exact_match']==''


        failed_exp_trace=trace_rows('experiment-'+failed_run['id'])[0];assert failed_exp_trace['status']=='failed';assert_trace_tree(failed_exp_trace)
        assert next(span for span in failed_exp_trace['spans'] if span['kind']=='model')['status']=='token_limit'
        try:api('evaluation/compare',dict(baseline_id=failed_run['id'],candidate_id=failed_run['id']));raise AssertionError('failed run eligible')
        except urllib.error.HTTPError as e:assert e.code==400
        run_path=data/'evaluation'/'runs'/(better['id']+'.json')
        original_receipt=run_path.read_bytes();tampered=json.loads(original_receipt);tampered['items'][0]['scores']['exact_match']=0
        run_path.write_text(json.dumps(tampered))
        try:api('evaluation/compare',dict(baseline_id=baseline['id'],candidate_id=better['id']));raise AssertionError('tampered metric receipt accepted')
        except urllib.error.HTTPError as e:assert e.code==400
        finally:run_path.write_bytes(original_receipt)
        tampered=json.loads(original_receipt);tampered['mean_scores']['exact_match']=0.123
        run_path.write_text(json.dumps(tampered))
        try:
            for endpoint,payload in [
                ('evaluation/compare',dict(baseline_id=baseline['id'],candidate_id=better['id'])),
                ('evaluation/matrices',dict(project_id=better['project_id'],baseline_id=baseline['id'],variants=[dict(label='Baseline',run_id=baseline['id']),dict(label='Candidate',run_id=better['id'])])),
            ]:
                try:api(endpoint,payload);raise AssertionError('tampered mean metric receipt accepted')
                except urllib.error.HTTPError as e:assert e.code==400
        finally:run_path.write_bytes(original_receipt)
        tampered=json.loads(original_receipt);tampered['items'][0]['output_truncated']=True
        run_path.write_text(json.dumps(tampered))
        try:api('evaluation/compare',dict(baseline_id=baseline['id'],candidate_id=better['id']));raise AssertionError('truncated output accepted as strict quality')
        except urllib.error.HTTPError as e:assert e.code==400
        finally:run_path.write_bytes(original_receipt)
        print('PASS immutable datasets, frozen experiment snapshots, per-item metrics, cancellation, output limits, tamper detection and conservative paired comparison')
        # Origin and intent protections.
        for headers in [{'Content-Type':'application/json'},{'Content-Type':'application/json','X-Allpaka-Client':'studio','Origin':'https://elsewhere.example'}]:
            try:api('sessions',settings,headers);raise AssertionError('unsafe request accepted')
            except urllib.error.HTTPError as e:assert e.code==403
        note=api('memory/notes',dict(project_id='default',name='Project decision',base_version=0,content='MEMORY_FIXTURE native decision'))
        duplicate_note=api('memory/notes',dict(project_id='default',name='Copied decision',base_version=0,content=note['content']))
        api('memory/notes',dict(project_id='global',name='Shared decision',base_version=0,content='MEMORY_FIXTURE global decision'))
        api('memory/notes',dict(project_id='default',name='Expired',base_version=0,content='MEMORY_FIXTURE outdated',expires_ms=1))
        reject('memory/notes',dict(id=note['id'],project_id='default',name='Stale',base_version=0,content='changed'),status=409)
        recalled_memory=create();act(recalled_memory,'send','TRY_MEMORY_RECALL')
        memory_state=wait(recalled_memory,lambda s:s['status']=='idle' and len(s['messages'])==4)
        memory_result=json.loads(memory_state['messages'][2]['content'])
        assert len(memory_result['matches'])==2 and all('outdated' not in row['memory']['content'] for row in memory_result['matches'])
        grouped=next(row for row in memory_result['matches'] if row['memory']['id']==note['id'])
        assert grouped['total_sources']==2 and {source['id'] for source in grouped['sources']}=={note['id'],duplicate_note['id']}
        assert memory_result['deduplication']=='exact_content'
        removed_note=api('memory/notes',dict(id=note['id'],project_id='default',name='Project decision',base_version=1,content=note['content'],removed=True))
        assert removed_note['version']==2 and api(f"memory/notes/{note['id']}/versions/1")==note
        expiry_calls=len(requests)
        due_note=api('memory/notes',dict(project_id='default',name='Upcoming deadline',base_version=0,content='Expiry review only',expires_ms=int(time.time()*1000)+86_400_000))
        expiry_before=api('memory/notes?project_id=default')
        expiry_report=Studio(base).memory_expiry('default',horizon_days=2)
        assert expiry_report['semantics']=='declared_expiry' and expiry_report['provider_calls']==0 and expiry_report['counts']==dict(expired=1,expiring=1,active=0,no_expiry=1,removed=1)
        assert all('content' not in row and row['project_id']=='default' for row in expiry_report['notes'])
        assert next(row for row in expiry_report['notes'] if row['id']==due_note['id'])['sha256']==due_note['sha256']
        assert Studio(base).memory_expiry('default',include_global=True,horizon_days=2)['counts']['no_expiry']==2
        assert Studio(base).memory_expiry('global',include_global=True,horizon_days=2)['counts']['no_expiry']==1
        assert Studio(base).memory_expiry('default',horizon_days=0)['counts']['expiring']==0
        reject('memory/expiry?project_id=missing-project')
        reject('memory/expiry?project_id=default&horizon_days=366')
        reject('memory/expiry?project_id=default&unexpected=true')
        assert api('memory/notes?project_id=default')==expiry_before and len(requests)==expiry_calls
        print('PASS HTTP/SDK memory expiry scope, metadata pins, horizon bounds and read-only behavior without inference',flush=True)
        print('PASS explicit project/global memory, native tool recall, expiry exclusion and immutable removal revisions')
        id=create();act(id,'send','hello');s=wait(id,lambda s:s['status']=='idle' and len(s['messages'])==2);assert 'hello' in s['messages'][-1]['content']
        consolidation_calls=len(requests)
        merge_first=api('memory/notes',dict(project_id='default',name='Source A',base_version=0,content='CONSOLIDATION_SOURCE_A'))
        merge_second=api('memory/notes',dict(project_id='default',name='Source B',base_version=0,content='CONSOLIDATION_SOURCE_B'))
        merge_pins=[{key:source[key] for key in ('id','version','sha256')} for source in (merge_first,merge_second)]
        merged_note=Studio(base).consolidate_memory('default',merge_pins,name='Reviewed merge',content='CONSOLIDATION_REVIEWED_CONTENT')
        assert merged_note['consolidation_sources']==merge_pins and merged_note['version']==1
        current_sources=Studio(base).memory_consolidation_source_status(merged_note['id'],1)
        assert current_sources['semantics']=='revision_status' and current_sources['provider_calls']==0 and current_sources['notes_modified'] is False
        assert [row['status'] for row in current_sources['sources']]==['current','current']
        assert 'CONSOLIDATION_SOURCE' not in json.dumps(current_sources)

        assert api(f"memory/notes/{merge_first['id']}/versions/1")==merge_first and api(f"memory/notes/{merge_second['id']}/versions/1")==merge_second
        merged_revision=api('memory/notes',dict(id=merged_note['id'],project_id='default',name='Reviewed merge revision',base_version=1,content='CONSOLIDATION_EDITED_CONTENT'))
        assert merged_revision['consolidation_sources']==merge_pins and merged_revision['version']==2
        api('memory/notes',dict(id=merge_first['id'],project_id='default',name='Source A revised',base_version=1,content='CONSOLIDATION_REVISED_SOURCE_A'))
        changed_sources=Studio(base).memory_consolidation_source_status(merged_note['id'],1)
        assert changed_sources['sources'][0]['status']=='changed' and changed_sources['sources'][0]['latest_version']==2 and changed_sources['sources'][0]['sha256']==merge_first['sha256']
        assert changed_sources['sources'][1]['status']=='current'
        removed_merge_second=api('memory/notes',dict(id=merge_second['id'],project_id='default',name=merge_second['name'],content=merge_second['content'],base_version=1,removed=True))
        removed_sources=Studio(base).memory_consolidation_source_status(merged_note['id'],1)
        assert [row['status'] for row in removed_sources['sources']]==['changed','removed'] and removed_sources['sources'][1]['latest_sha256']==removed_merge_second['sha256']
        assert Studio(base).memory_consolidation_source_status(merge_first['id'],1)['sources']==[]
        reject('memory/notes',dict(project_id='default',name='Stale merge',base_version=0,content='merged',consolidation_sources=merge_pins))
        reject('memory/notes',dict(id=merged_note['id'],project_id='default',name='Changed origin',base_version=2,content='merged',consolidation_sources=list(reversed(merge_pins))))
        assert api(f"memory/notes/{merged_note['id']}/versions/1")==merged_note and len(requests)==consolidation_calls
        print('PASS HTTP reviewed memory consolidation pins hashes/versions, preserves sources and origin, rejects stale sources without inference',flush=True)
        synthesis_sources=[api('memory/notes',dict(project_id='default',base_version=0,name=name,content=content)) for name,content in [('Synthesis A','MODEL_SOURCE_A qualified'),('Synthesis B','MODEL_SOURCE_B retained')]]
        synthesis_pins=[{key:row[key] for key in ('id','version','sha256')} for row in synthesis_sources]
        capacity_directory=data/'memory'/'proposals';capacity_directory.mkdir(parents=True,exist_ok=True)
        capacity_original=set(capacity_directory.glob('*.json'));capacity_calls=len(requests);capacity_notes=api('memory/notes?project_id=default')
        capacity_files=[]
        try:
            for index in range(999-len(capacity_original)):
                path=capacity_directory/('capacity-'+str(index)+'.json');path.write_text('{}');capacity_files.append(path)
            for path,body in [('memory/consolidation-proposals',dict(settings=settings,sources=synthesis_pins)),(f'sessions/{id}/memory-proposals',dict(settings=settings,message_count=2))]:
                reject(path,body)
            assert len(requests)==capacity_calls and api('memory/notes?project_id=default')==capacity_notes
        finally:
            for path in capacity_files:path.unlink()
        assert set(capacity_directory.glob('*.json'))==capacity_original
        capacity_files=[]
        try:
            for index in range(130):
                path=capacity_directory/('capacity-bytes-'+str(index)+'.json')
                with path.open('wb') as file:file.truncate(128000)
                capacity_files.append(path)
            reject('memory/consolidation-proposals',dict(settings=settings,sources=synthesis_pins))
            reject(f'sessions/{id}/memory-proposals',dict(settings=settings,message_count=2))
            assert len(requests)==capacity_calls and api('memory/notes?project_id=default')==capacity_notes
        finally:
            for path in capacity_files:path.unlink()
        assert set(capacity_directory.glob('*.json'))==capacity_original
        print('PASS HTTP proposal count/byte capacity refuses extraction and consolidation before provider calls, without changing notes or retained proposals',flush=True)
        before_synthesis_notes=api('memory/notes?project_id=default');synthesis_calls=len(requests)
        synthesis_proposal=Studio(base).propose_memory_consolidation(settings,synthesis_pins)
        assert synthesis_proposal['kind']=='memory_consolidation_proposal' and synthesis_proposal['consolidation_sources']==synthesis_pins and synthesis_proposal['accepted'] is False
        assert api('memory/notes?project_id=default')==before_synthesis_notes and len(requests)==synthesis_calls+1
        assert Studio(base).memory_proposal(synthesis_proposal['id'])==synthesis_proposal
        synthesis_trace=trace_rows('memory-consolidation-'+synthesis_proposal['id'])[0]
        assert synthesis_trace['status']=='completed' and len(synthesis_trace['spans'])==2 and 'MODEL_SOURCE_' not in json.dumps(synthesis_trace) and 'Combined fact' not in json.dumps(synthesis_trace)
        model_source=json.loads(requests[-1]['messages'][-1]['content']);assert len(model_source)==2 and model_source[0]['sha256']==synthesis_sources[0]['sha256'] and model_source[0]['content']==synthesis_sources[0]['content']
        synthesis_accepted=Studio(base).accept_memory(synthesis_proposal['id'],0,content='Explicitly reviewed synthesis')
        assert synthesis_accepted['consolidation_sources']==synthesis_pins and synthesis_accepted['proposal_source']==dict(proposal_id=synthesis_proposal['id'],note_index=0)
        reject('memory/notes',dict(project_id='default',name='Missing pins',content='reviewed',base_version=0,proposal_source=dict(proposal_id=synthesis_proposal['id'],note_index=0)))
        synthesis_input_policy=Studio(base).save_guardrail_policy([dict(id='synthesis-source',kind='forbidden_substrings',value=['MODEL_SOURCE_A'])])['policy']['policy_sha256']
        synthesis_output_policy=Studio(base).save_guardrail_policy([dict(id='synthesis-result',kind='forbidden_substrings',value=['Combined fact'])])['policy']['policy_sha256']
        synthesis_settings=dict(settings,guardrails=dict(input_policy_sha256=synthesis_input_policy,output_policy_sha256=policy_hash,action='block'))
        guard_synthesis_calls=len(requests);guard_synthesis_saved=len(list((data/'memory'/'proposals').glob('*.json')));guard_synthesis_notes=api('memory/notes?project_id=default')
        reject('memory/consolidation-proposals',dict(settings=synthesis_settings,sources=synthesis_pins))
        assert len(requests)==guard_synthesis_calls and len(list((data/'memory'/'proposals').glob('*.json')))==guard_synthesis_saved
        synthesis_settings['guardrails']=dict(input_policy_sha256=policy_hash,output_policy_sha256=synthesis_output_policy,action='block')
        reject('memory/consolidation-proposals',dict(settings=synthesis_settings,sources=synthesis_pins))
        assert len(requests)==guard_synthesis_calls+1 and len(list((data/'memory'/'proposals').glob('*.json')))==guard_synthesis_saved
        synthesis_settings['guardrails']['action']='observe'
        observed_synthesis=Studio(base).propose_memory_consolidation(synthesis_settings,synthesis_pins)
        assert observed_synthesis['accepted'] is False and len(requests)==guard_synthesis_calls+2 and api('memory/notes?project_id=default')==guard_synthesis_notes
        observed_synthesis_trace=trace_rows('memory-consolidation-'+observed_synthesis['id'])[0]
        assert any(span['usage'].get('guardrail_receipt',{}).get('passed') is False for span in observed_synthesis_trace['spans'])
        assert 'MODEL_SOURCE_' not in json.dumps(observed_synthesis_trace) and 'Combined fact' not in json.dumps(observed_synthesis_trace)
        print('PASS HTTP consolidation guards block input before inference, withhold rejected output and retain observed candidate without auto-save',flush=True)
        consolidation_catalog_calls=len(requests);consolidation_catalog=Studio(base).memory_consolidation_proposals('default',limit=1)
        assert consolidation_catalog['kind']=='memory_consolidation_catalog' and consolidation_catalog['total']==2 and consolidation_catalog['has_more'] and len(consolidation_catalog['proposals'])==1
        next_consolidation_page=Studio(base).memory_consolidation_proposals('default',offset=1,limit=1)
        assert not next_consolidation_page['has_more'] and len(next_consolidation_page['proposals'])==1
        assert {consolidation_catalog['proposals'][0]['id'],next_consolidation_page['proposals'][0]['id']}=={synthesis_proposal['id'],observed_synthesis['id']}
        assert all(row['source_count']==2 and row['note_count']==1 and 'notes' not in row and 'settings' not in row for row in consolidation_catalog['proposals']+next_consolidation_page['proposals'])
        assert Studio(base).memory_consolidation_proposals('global')['total']==0 and len(requests)==consolidation_catalog_calls
        reject('memory/consolidation-proposals?project_id=default&limit=101');reject('memory/consolidation-proposals?project_id=default&unexpected=true')

        synthesis_bad=api('memory/notes',dict(id=synthesis_sources[0]['id'],project_id='default',name='Malformed synthesis source',base_version=1,content='BAD_CONSOLIDATE'))
        before_bad_synthesis=len(requests);reject('memory/consolidation-proposals',dict(settings=settings,sources=synthesis_pins));assert len(requests)==before_bad_synthesis
        bad_pins=[{key:synthesis_bad[key] for key in ('id','version','sha256')},synthesis_pins[1]]
        saved_proposals=len(list((data/'memory'/'proposals').glob('*.json')))
        reject('memory/consolidation-proposals',dict(settings=settings,sources=bad_pins))
        assert len(requests)==before_bad_synthesis+1 and len(list((data/'memory'/'proposals').glob('*.json')))==saved_proposals
        print('PASS HTTP model consolidation source pins, proposal-only generation, metadata trace, explicit acceptance, stale preflight and malformed candidate rejection',flush=True)
        notes_before=api('memory/notes?project_id=default')
        proposal=api(f'sessions/{id}/memory-proposals',dict(settings=settings,message_count=2))
        assert not proposal['accepted'] and proposal['message_count']==2 and len(proposal['source_sha256'])==64 and len(proposal['notes'])==1
        assert api('memory/proposals/'+proposal['id'])==proposal
        extraction_status_calls=len(requests);extraction_status_notes=api('memory/notes?project_id=default')
        extraction_status=Studio(base).memory_extraction_source_status(proposal['id'])
        assert extraction_status['kind']=='memory_extraction_source_status' and extraction_status['status']=='current' and extraction_status['semantics']=='extraction_prefix'
        assert extraction_status['source_sha256']==proposal['source_sha256']==extraction_status['current_source_sha256'] and extraction_status['message_count']==2
        assert extraction_status['provider_calls']==0 and extraction_status['notes_modified'] is False and len(requests)==extraction_status_calls
        assert 'content' not in extraction_status and api('memory/notes?project_id=default')==extraction_status_notes
        reject('memory/proposals/'+synthesis_proposal['id']+'/source-status')
        print('PASS extraction prefix status reports exact hash without source text, provider calls or note mutation',flush=True)

        extraction_status_fixtures=[]
        for label in ['changed','unavailable']:
            source_session=create();act(source_session,'send','SOURCE_STATUS_'+label)
            wait(source_session,lambda state:state['status']=='idle' and len(state['messages'])==2)
            retained=api(f'sessions/{source_session}/memory-proposals',dict(settings=settings,message_count=2))
            act(source_session,'send','APPENDED_SOURCE_STATUS')
            wait(source_session,lambda state:state['status']=='idle' and len(state['messages'])==4)
            calls=len(requests);appended=Studio(base).memory_extraction_source_status(retained['id'])
            assert appended['status']=='current' and appended['current_message_count']==4 and len(requests)==calls
            extraction_status_fixtures.append((label,source_session,retained))
        reject('memory/proposals/missing',status=404)
        catalog=api(f'sessions/{id}/memory-proposals')['proposals']
        assert len(catalog)==1 and catalog[0]['id']==proposal['id'] and catalog[0]['note_count']==1
        assert 'notes' not in catalog[0] and 'settings' not in catalog[0]
        assert api('sessions/other-conversation/memory-proposals')['proposals']==[]
        assert api('memory/notes?project_id=default')==notes_before
        assert json.loads((data/'memory'/'proposals'/f"{proposal['id']}.json").read_text())==proposal
        extraction_trace=trace_rows('memory-extraction-'+proposal['id'])[0]
        assert extraction_trace['status']=='completed' and len(extraction_trace['spans'])==2
        assert 'proposed fact' not in json.dumps(extraction_trace)
        memory_input_policy=Studio(base).save_guardrail_policy([dict(id='source',kind='forbidden_substrings',value=['hello'])])['policy']['policy_sha256']
        memory_output_policy=Studio(base).save_guardrail_policy([dict(id='proposal',kind='forbidden_substrings',value=['notes'])])['policy']['policy_sha256']
        memory_guards=dict(settings,guardrails=dict(input_policy_sha256=memory_input_policy,output_policy_sha256=policy_hash,action='block'))
        before_memory_checks=len(requests)
        before_proposals=len(api(f'sessions/{id}/memory-proposals')['proposals'])
        reject(f'sessions/{id}/memory-proposals',dict(settings=memory_guards,message_count=2))
        assert len(requests)==before_memory_checks
        memory_guards['guardrails']=dict(input_policy_sha256=policy_hash,output_policy_sha256=memory_output_policy,action='block')
        reject(f'sessions/{id}/memory-proposals',dict(settings=memory_guards,message_count=2))
        assert len(requests)==before_memory_checks+1 and len(api(f'sessions/{id}/memory-proposals')['proposals'])==before_proposals
        memory_guards['guardrails']['action']='observe'
        observed_proposal=api(f'sessions/{id}/memory-proposals',dict(settings=memory_guards,message_count=2))
        assert not observed_proposal['accepted'] and observed_proposal['notes']
        assert api('memory/notes?project_id=default')==notes_before
        print('PASS memory extraction policies input no inference, blocked proposal not saved and observe without auto-accept',flush=True)

        reject(f'sessions/{id}/memory-proposals',dict(settings=settings,message_count=3))
        malformed=create();act(malformed,'send','BAD_EXTRACT');wait(malformed,lambda s:s['status']=='idle' and len(s['messages'])==2)
        reject(f'sessions/{malformed}/memory-proposals',dict(settings=settings,message_count=2))
        assert api('memory/notes?project_id=default')==notes_before
        source=dict(proposal_id=proposal['id'],note_index=0)
        accepted=api('memory/notes',dict(project_id='default',name='Reviewed fact',base_version=0,content='Human edited fact',proposal_source=source))
        assert accepted['proposal_source']==source
        updated=api('memory/notes',dict(id=accepted['id'],project_id='default',name='Updated fact',base_version=1,content='Revised fact'))
        assert updated['proposal_source']==source
        reject('memory/notes',dict(id=accepted['id'],project_id='default',name='Changed origin',base_version=2,content='fact',proposal_source=dict(proposal_id='missing',note_index=0)))
        reject('memory/notes',dict(project_id='global',name='Wrong scope',base_version=0,content='fact',proposal_source=source))
        reject('memory/notes',dict(project_id='default',name='Missing source',base_version=0,content='fact',proposal_source=dict(proposal_id=proposal['id'],note_index=10)))
        sdk=Studio(base)
        assert sdk.memory_proposal(proposal['id'])==proposal
        assert sdk.memory_proposals(id)['proposals']==api(f'sessions/{id}/memory-proposals')['proposals']
        sdk_note=sdk.accept_memory(proposal['id'],0,content='SDK reviewed fact')
        assert sdk_note['proposal_source']==source and sdk_note['content']=='SDK reviewed fact'
        print('PASS bounded memory proposals and explicit reviewed notes with immutable verified provenance')
        print('PASS chat, model catalog, usage, request guards')
        traces=api('observability/traces?session_id='+id)['traces']
        assert len(traces)==1 and traces[0]['status']=='completed'
        assert [span['kind'] for span in traces[0]['spans']]==['turn','model']
        assert traces[0]['spans'][1]['parent_id']==0 and traces[0]['spans'][1]['usage']['prompt_tokens']==3
        assert traces[0]['spans'][1]['provider_id']=='local'
        provider_summary=Studio(base).trace_summary(session_id=id)
        assert any(group['provider_id']=='local' and group['source']=='native' for group in provider_summary['models'])
        assert all(span['duration_ms'] is not None for span in traces[0]['spans'])
        assert 'hello' not in json.dumps(traces)
        print('PASS native model trace tree, timings, reported tokens and content privacy')
        queue_calls=len(requests)
        queue=Studio(base).create_review_queue('default','Review fixture',[dict(trace_id=traces[0]['id'],span_id=1)],instructions='Check accuracy')
        assert queue['version']==1 and queue['instructions']=='Check accuracy' and api('observability/review-queues/'+queue['id'])==queue
        assert Studio(base).review_queue(queue['id'])==queue
        assigned_queue=Studio(base).assign_review_queue(queue['id'],0,'reviewer-a',base_version=1)
        assert assigned_queue['version']==2 and assigned_queue['assignments']=={'0':'reviewer-a'} and Studio(base).review_queue(queue['id'])==assigned_queue
        try:Studio(base).assign_review_queue(queue['id'],0,'reviewer-b',base_version=1);raise AssertionError('stale assignment overwrote reviewer')
        except EvaluationError as error:assert error.reason=='http_409'
        queue=Studio(base).assign_review_queue(queue['id'],0,None,base_version=2)
        assert queue['version']==3 and queue['assignments']=={}
        queue_page=Studio(base).review_queues('default',limit=1);assert queue_page['total']==1 and queue_page['queues'][0]['id']==queue['id'] and 'instructions' not in queue_page['queues'][0]
        api('projects',dict(id='feed-0001',name='Foreign review project',instructions='',roots=[dict(alias='workspace',path=str(workspace),writable=False)]))
        reject('observability/review-queues',dict(project_id='feed-0001',name='Wrong ownership',targets=[dict(trace_id=traces[0]['id'])]))
        assert Studio(base).review_queues('feed-0001')['total']==0
        reject('observability/review-queues?project_id=default&limit=101')
        reject('observability/review-queues',dict(project_id='default',name='Duplicate',targets=[dict(trace_id=traces[0]['id'])]*2))
        reject('observability/review-queues',dict(project_id='default',name='Missing',targets=[dict(trace_id='trace-nonexistent')]))
        reject('observability/review-queues',dict(project_id='default',name='Invalid span',targets=[dict(trace_id=traces[0]['id'],span_id=999)]))
        assert len(requests)==queue_calls
        print('PASS durable review queue source creation/detail and duplicate/missing/span admission checks without inference',flush=True)
        feedback_endpoint='observability/traces/'+traces[0]['id']+'/feedback'
        assert Studio(base).feedback(traces[0]['id'])['version']==0
        review=dict(author='reviewer-a',span_id=1,metric='accuracy',value=0.75,comment='<script>literal feedback</script>',correction='corrected answer')
        receipt=Studio(base).save_feedback(traces[0]['id'],review,base_version=0);assert receipt['version']==1 and receipt['summaries'][0]['mean']==0.75
        saved_review=receipt['annotations'][0]
        review_completion_calls=len(requests)
        queue=Studio(base).assign_review_queue(queue['id'],0,'reviewer-b',base_version=3)
        completion_endpoint='observability/review-queues/'+queue['id']+'/completion'
        completion=dict(base_version=4,target_index=0,action='complete',feedback_version=1,annotation_id=saved_review['id'])
        reject(completion_endpoint,completion)
        queue=Studio(base).assign_review_queue(queue['id'],0,'reviewer-a',base_version=4)
        completed_queue=Studio(base).complete_review_queue(queue['id'],0,1,saved_review['id'],base_version=5)
        assert completed_queue['version']==6 and completed_queue['completions']['0']==dict(reviewer='reviewer-a',feedback_version=1,annotation_id=saved_review['id'])
        reject('observability/review-queues/'+queue['id']+'/assignments',dict(base_version=6,target_index=0,reviewer='reviewer-b'))
        reject(completion_endpoint,dict(completion,base_version=6))
        reopened_queue=Studio(base).reopen_review_queue(queue['id'],0,base_version=6)
        assert reopened_queue['version']==7 and reopened_queue['completions']=={} and reopened_queue['assignments']['0']=='reviewer-a'
        completed_queue=Studio(base).complete_review_queue(queue['id'],0,1,saved_review['id'],base_version=7)
        assert completed_queue['version']==8 and len(requests)==review_completion_calls
        print('PASS queue completion pins target/reviewer feedback evidence, blocks reassignment, reopens explicitly and never invokes provider',flush=True)
        for bad in [dict(review,span_id=999),dict(review,metric=None),dict(review,value=1000001),dict(review,value=None,category='')]:
            try:api(feedback_endpoint,dict(base_version=1,annotation=bad));raise AssertionError('invalid feedback accepted')
            except urllib.error.HTTPError as e:assert e.code==400
        try:api(feedback_endpoint,dict(base_version=0,annotation=review));raise AssertionError('stale feedback overwrote another review')
        except urllib.error.HTTPError as e:assert e.code==409
        receipt=api(feedback_endpoint,dict(base_version=1,annotation=dict(author='reviewer-b',span_id=1,metric='accuracy',value=0.25)))
        assert receipt['summaries'][0]['mean']==0.5 and receipt['summaries'][0]['count']==2
        receipt=api(feedback_endpoint,dict(base_version=2,annotation=dict(author='reviewer-a',metric='acceptance',category='needs-work')))
        assert any(score['categories']=={'needs-work':1} for score in receipt['summaries'])
        receipt=api(feedback_endpoint,dict(base_version=3,annotation=dict(saved_review,deleted=True)))
        assert next(score for score in receipt['summaries'] if score['metric']=='accuracy')['mean']==0.25
        reopen_deleted=Studio(base).reopen_review_queue(queue['id'],0,base_version=8)
        assert reopen_deleted['version']==9
        reject(completion_endpoint,dict(completion,base_version=9,feedback_version=4))
        reject(completion_endpoint,dict(completion,base_version=9,feedback_version=3,annotation_id=receipt['annotations'][-1]['id']))
        reject(completion_endpoint,dict(completion,base_version=9,annotation_id='missing-annotation'))
        try:Studio(base).complete_review_queue(queue['id'],0,1,saved_review['id'],base_version=8);raise AssertionError('stale completion accepted')
        except EvaluationError as error:assert error.reason=='http_409'
        completed_queue=Studio(base).complete_review_queue(queue['id'],0,1,saved_review['id'],base_version=9)
        assert completed_queue['version']==10 and completed_queue['completions']['0']['feedback_version']==1
        archive_calls=len(requests);client=Studio(base)
        archived_queue=client.set_review_queue_archived(queue['id'],True,base_version=10)
        assert archived_queue['version']==11 and archived_queue['archived'] and archived_queue['targets']==completed_queue['targets'] and archived_queue['completions']==completed_queue['completions']
        assert client.review_queues('default',archived=False)['total']==0
        archive_page=client.review_queues('default',archived=True,status='completed',reviewer='reviewer-a',name='REVIEW')
        assert archive_page['total']==1 and archive_page['queues'][0]['archived'] and archive_page['filters']==dict(archived=True,status='completed',reviewer='reviewer-a',name='REVIEW')
        assert client.review_queues('feed-0001',archived=True)['total']==0
        reject('observability/review-queues/'+queue['id']+'/assignments',dict(base_version=11,target_index=0,reviewer='other'))
        reject(completion_endpoint,dict(base_version=11,target_index=0,action='reopen'))
        reject('observability/review-queues/'+queue['id']+'/lifecycle',dict(base_version=11,archived=True))
        reject('observability/review-queues/'+queue['id']+'/lifecycle',dict(base_version=10,archived=False),status=409)
        archived_csv=list(csv.DictReader(io.StringIO(client.export_review_queue_csv(queue['id']))))
        assert archived_csv[0]['archived']=='true' and archived_csv[0]['annotation_id']==saved_review['id'] and archived_csv[0]['queue_version']=='11'
        restored=client.set_review_queue_archived(queue['id'],False,base_version=11)
        assert restored['version']==12 and not restored['archived'] and restored['completions']==completed_queue['completions']
        completed_queue=client.set_review_queue_archived(queue['id'],True,base_version=12)
        assert completed_queue['version']==13 and len(requests)==archive_calls
        queue_history=client.review_queue_history(queue['id'],limit=2)
        assert queue_history['history_complete'] and queue_history['total']==13 and queue_history['has_more'] and [e['version'] for e in queue_history['entries']]==[13,12]
        assert [e['action'] for e in queue_history['entries']]==['archive','restore']
        history_tail=client.review_queue_history(queue['id'],offset=12,limit=2)
        assert history_tail['entries'][0]['action']=='create' and not history_tail['has_more']
        completion_history=client.review_queue_history(queue['id'],offset=3,limit=1)['entries'][0]
        assert completion_history['action']=='complete' and completion_history['feedback_version']==1 and completion_history['annotation_id']==saved_review['id']
        reject('observability/review-queues/'+queue['id']+'/history?limit=101')
        print('PASS HTTP review queue archive/restore, combined catalog filters, archived write rejection and CSV evidence without inference',flush=True)
        receipt=api(feedback_endpoint,dict(base_version=4,annotation=saved_review))
        assert next(score for score in receipt['summaries'] if score['metric']=='accuracy')['mean']==0.5
        assert Studio(base).feedback(traces[0]['id'],version=1)['annotations']==[saved_review]
        history_calls=len(requests);history=api(feedback_endpoint+'/versions?offset=0&limit=2')
        assert history['total']==5 and history['has_more'] and [row['version'] for row in history['versions']]==[5,4]
        assert history['versions'][0]['active_count']==3 and history['versions'][1]['active_count']==2 and history['provider_calls']==0
        assert 'corrected answer' not in json.dumps(history) and 'reviewer-a' not in json.dumps(history)
        assert [row['version'] for row in api(feedback_endpoint+'/versions?offset=2&limit=2')['versions']]==[3,2]
        reject(feedback_endpoint+'/versions?limit=101');assert len(requests)==history_calls
        assert 'corrected answer' not in json.dumps(api('observability/traces?session_id='+id))
        try:api('observability/traces/trace-nonexistent/feedback',dict(base_version=0,annotation=review));raise AssertionError('feedback for missing trace accepted')
        except urllib.error.HTTPError as e:assert e.code==400
        print('PASS trace/span numeric and categorical feedback, comments, corrections, revision conflicts, removal/restore and content isolation')

        first_mark=api(f'sessions/{id}/bookmarks',dict(message_index=0,label='  BOOKMARK_PRIVATE <script>literal</script>  '))['bookmarks'][0]
        assert first_mark['label']=='BOOKMARK_PRIVATE <script>literal</script>' and first_mark['created_ms']>0
        changed=api(f'sessions/{id}/bookmarks',dict(message_index=0,label=first_mark['label']))['bookmarks'][0]
        assert changed==first_mark
        marks=api(f'sessions/{id}/bookmarks',dict(message_index=1,label='Answer landmark'))['bookmarks']
        reject(f'sessions/{id}/bookmarks',dict(message_index=2,label='missing'))
        reject(f'sessions/{id}/bookmarks',dict(message_index=0,label=' '))
        reject(f'sessions/{id}/bookmarks',dict(message_index=0,label='x'*201))
        assert api('sessions/'+id)['bookmarks']==marks
        matching=api('sessions?bookmarked=true&q=BOOKMARK_PRIVATE')
        assert any(row['id']==id and row['bookmarks']==2 for row in matching)
        assert all(row['bookmarks']>0 for row in matching)
        marked_copy=api('sessions/import',dict(session=api('sessions/'+id),project_id='default'))['id']
        assert api('sessions/'+marked_copy)['bookmarks']==marks
        invalid_marks=api('sessions/'+id);invalid_marks['bookmarks']=marks+[marks[0]]
        reject('sessions/import',dict(session=invalid_marks,project_id='default'))
        branch=api(f'sessions/{id}/branch',dict(message_count=2))['id']
        branched=api('sessions/'+branch)
        assert branched['messages']==s['messages'] and branched['parent']==dict(session_id=id,message_count=2)
        assert branched['queue']==[] and branched['status']=='idle' and branched['bookmarks']==marks
        act(branch,'send','BRANCH-ONLY');wait(branch,lambda x:x['status']=='idle' and len(x['messages'])==4)
        assert len(api('sessions/'+id)['messages'])==2
        empty=api(f'sessions/{id}/branch',dict(message_count=0))['id'];assert api('sessions/'+empty)['messages']==[] and api('sessions/'+empty)['bookmarks']==[]
        for count in [1,3]:
            try:api(f'sessions/{id}/branch',dict(message_count=count));raise AssertionError('invalid boundary accepted')
            except urllib.error.HTTPError as e:assert e.code==400
        assert 'BOOKMARK_PRIVATE' not in json.dumps(requests)
        remove_request=urllib.request.Request(base+f'/api/sessions/{branch}/bookmarks/1',method='DELETE',headers={'X-Allpaka-Client':'studio'})
        with urllib.request.urlopen(remove_request) as response:assert len(json.load(response)['bookmarks'])==1
        assert api('sessions/'+id)['bookmarks']==marks
        print('PASS bookmark persistence, bounded labels/targets, independent branch copies, inert imports and model-context isolation')
        print('PASS independent branches, parent provenance, safe history boundaries')
        # Queue preserves order; steer is applied before the queued task.
        act(id,'send','SLOW');wait(id,lambda s:s['status']=='running' and s['messages'][-1]['content'].startswith('started'))
        try:api(f'sessions/{id}/branch',dict(message_count=2));raise AssertionError('running branch accepted')
        except urllib.error.HTTPError as e:assert e.code==409
        act(id,'send','QUEUED');act(id,'steer','STEER-UPDATED')
        s=wait(id,lambda s:s['status']=='idle' and not s['queue'] and any(m['role']=='assistant' and 'QUEUED' in m['content'] for m in s['messages']))
        users=[m['content'] for m in s['messages'] if m['role']=='user'];assert next(i for i,x in enumerate(users) if 'STEER-UPDATED' in x)<users.index('QUEUED')
        act(id,'send','SLOW');wait(id,lambda s:s['status']=='running' and s['messages'][-1]['content'].startswith('started'))
        before=time.time();act(id,'send_now','IMMEDIATE');s=wait(id,lambda s:s['status']=='idle' and 'IMMEDIATE' in s['messages'][-1]['content']);assert time.time()-before<1
        print('PASS queue, steer, immediate cancellation')
        # Stop retains queued messages; resume drains them without lost input.
        act(id,'send','SLOW');wait(id,lambda s:s['status']=='running');act(id,'send','AFTER-STOP');act(id,'stop');s=wait(id,lambda s:s['status']=='paused');assert s['queue'][0]['text']=='AFTER-STOP'
        act(id,'resume');wait(id,lambda s:s['status']=='idle' and 'AFTER-STOP' in s['messages'][-1]['content'])
        print('PASS stop and resume')
        traces=api('observability/traces?session_id='+id)['traces']
        assert any(trace['status']=='interrupted' for trace in traces)
        print('PASS cancellation marks interrupted traces')
        read_only_auto=create(mode='auto',allow_writes=True);act(read_only_auto,'send','CHECK-CAPABILITIES')
        wait(read_only_auto,lambda s:s['status']=='idle' and len(s['messages'])==2)
        tool_names=[t['function']['name'] for t in requests[-1]['tools']]
        assert 'read_file' in tool_names and 'write_file' not in tool_names and 'edit_file' not in tool_names
        assert 'Tools actually available for THIS request:' in requests[-1]['messages'][0]['content']
        project=api('projects',dict(id='default',name='Test project',instructions='',roots=[dict(alias='workspace',path=str(workspace),writable=True),dict(alias='reference',path=str(reference),writable=False)]))
        planned=create(mode='plan',allow_writes=True);act(planned,'send','TRY_WRITE');s=wait(planned,lambda s:s['status']=='idle' and len(s['messages'])>=4);assert not (workspace/'result.txt').exists();assert 'requires Auto' in next(m['content'] for m in s['messages'] if m['role']=='tool')
        milestone_manual=create(mode='goal');before_milestone_calls=len(requests)
        assert Studio(base).session_plan(milestone_manual)['revision']==0
        saved_milestones=Studio(base).update_session_plan(milestone_manual,[dict(title='Check fixture',status='in_progress',acceptance=['Fixture passes'],evidence=[])],base_revision=0)
        manual_step=saved_milestones['steps'][0];assert manual_step['id'] and saved_milestones['revision']==1
        checkpoint_before=Studio(base).session_plan(milestone_manual)
        invalid_completion=dict(manual_step,status='completed')
        try:Studio(base).update_session_plan(milestone_manual,[invalid_completion],base_revision=1);raise AssertionError('milestone completion without evidence accepted')
        except EvaluationError as error:assert error.reason=='http_409'
        assert Studio(base).session_plan(milestone_manual)==checkpoint_before
        completed_manual_step=dict(manual_step,status='completed',evidence=['Reported fixture result'])
        Studio(base).update_session_plan(milestone_manual,[completed_manual_step],base_revision=1)
        try:Studio(base).update_session_plan(milestone_manual,[completed_manual_step],base_revision=1);raise AssertionError('stale plan edit accepted')
        except EvaluationError as error:assert error.reason=='http_409'
        try:Studio(base).update_session_plan(milestone_manual,[dict(completed_manual_step,status='pending')],base_revision=2);raise AssertionError('completed Goal reopened silently')
        except EvaluationError as error:assert error.reason=='http_409'
        Studio(base).update_session_plan(milestone_manual,[dict(completed_manual_step,status='pending')],base_revision=2,allow_reopen=True)
        manual_plan=Studio(base).session_plan(milestone_manual);assert manual_plan['revision']==3 and manual_plan['checkpoints'][1]['steps'][0]['status']=='completed'
        assert manual_plan['evidence_basis']=='reported' and not manual_plan['automatic_replay'] and len(requests)==before_milestone_calls
        print('PASS durable milestone IDs, criteria/reported evidence, revision conflicts and explicit human reopen without inference',flush=True)
        goal_continued=create(mode='goal',max_steps=6);before_goal_continue=len(requests)
        act(goal_continued,'send','GOAL_CONTINUE_FIXTURE')
        continued=wait(goal_continued,lambda state:state['status']=='idle' and state['plan_revision']==2)
        continued_plan=Studio(base).session_plan(goal_continued)
        assert len(requests)==before_goal_continue+4 and continued_plan['reported_completion_ready']
        assert continued_plan['steps'][0]==continued_plan['checkpoints'][0]['steps'][0]
        assert sum(message['content'].startswith('Goal continuation:') for message in continued['messages'] if message['role']=='user')==1
        assert trace_rows(goal_continued)[0]['status']=='completed'
        first_goal=continued_plan['goal'];old_goal_plan=continued_plan['steps']
        assert first_goal['message_index']==0 and first_goal['id']
        before_new_goal=len(requests);act(goal_continued,'send','GOAL_STUCK_FIXTURE')
        next_goal=wait(goal_continued,lambda state:state['status']=='paused' and state.get('goal',{}).get('id')!=first_goal['id'])
        next_goal_packet=Studio(base).session_plan(goal_continued)
        assert len(requests)==before_new_goal+2 and next_goal_packet['steps']==[] and not next_goal_packet['reported_completion_ready']
        assert next_goal_packet['checkpoints'][-2]['steps']==old_goal_plan and next_goal_packet['checkpoints'][-2]['goal_id']==first_goal['id']
        assert next_goal_packet['checkpoints'][-1]['goal_id']==next_goal['goal']['id']
        assert next_goal_packet['checkpoints'][-2]['goal_origin']==first_goal
        assert next_goal_packet['checkpoints'][-1]['goal_origin']==next_goal['goal']
        before_new_resume=len(requests);act(goal_continued,'resume')
        resumed_goal=wait(goal_continued,lambda state:state['status']=='paused' and len(state['messages'])>len(next_goal['messages']))
        assert len(requests)==before_new_resume+2 and resumed_goal['goal']==next_goal['goal']
        print('PASS new user Goal cannot inherit prior completion; old checkpoints remain attributable and explicit resume preserves the current objective',flush=True)
        goal_stuck=create(mode='goal',max_steps=6);before_goal_stuck=len(requests)
        act(goal_stuck,'send','GOAL_STUCK_FIXTURE');stuck=wait(goal_stuck,lambda state:state['status']=='paused')
        assert len(requests)==before_goal_stuck+2 and 'Цель не завершена' in stuck['notice']
        assert not Studio(base).session_plan(goal_stuck)['reported_completion_ready']
        assert trace_rows(goal_stuck)[0]['status']=='goal_incomplete'
        print('PASS Goal completion gate continues unfinished milestones within the step budget, preserves completed IDs and pauses repeated unsupported completion without runaway calls',flush=True)
        auto=create(mode='auto',allow_writes=True);act(auto,'send','TRY_WRITE');wait(auto,lambda s:s['status']=='idle' and len(s['messages'])>=4);assert (workspace/'result.txt').read_text()=='verified write'
        loop_session=create(mode='goal',allow_writes=True,max_steps=6);act(loop_session,'send','NO_PROGRESS_LOOP')
        stalled=wait(loop_session,lambda s:s['status']=='paused')
        assert 'Три одинаковых' in stalled['notice'] and not (workspace/'blocked-loop.txt').exists()
        loop_tools=[message for message in stalled['messages'] if message['role']=='tool']
        assert len(loop_tools)==4 and all('error' in message['content'] for message in loop_tools[:3]) and 'not executed' in loop_tools[3]['content']
        assert trace_rows(loop_session)[0]['status']=='no_progress'
        before_summary=len(requests)
        summary=Studio(base).trace_summary(session_id=loop_session)
        assert summary['trace_count']==1 and summary['model_calls']==1 and summary['cost_unknown_calls']==1 and summary['input_tokens_unknown_calls']==1
        assert summary['trace_statuses']['no_progress']==1 and summary['price_estimates'] is False
        failed_scope=Studio(base).trace_summary(session_id=loop_session,status='no_progress');assert failed_scope['trace_count']==1 and failed_scope['status']=='no_progress'
        empty_status=Studio(base).trace_summary(session_id=loop_session,status='completed');assert empty_status['trace_count']==0 and empty_status['conversations']==[]
        reject('observability/summary?status=INVALID')
        conversation=summary['conversations'][0];assert summary['conversation_count']==1 and not summary['conversations_truncated']
        assert conversation['session_id']==loop_session and conversation['project_id']=='default' and conversation['trace_count']==1
        assert conversation['model_usage']['calls']==1 and conversation['model_usage']['input_tokens_unknown_calls']==1 and conversation['trace_statuses']['no_progress']==1
        drill=api('observability/traces?'+urllib.parse.urlencode(dict(project_id='default',session_id=loop_session,offset=0,limit=20,since_ms=conversation['first_started_ms'],until_ms=conversation['last_started_ms'])))
        assert drill['total']==1 and all(row['project_id']=='default' and row['session_id']==loop_session for row in drill['traces'])
        page=Studio(base).trace_summary(session_id=loop_session,conversation_offset=1,conversation_limit=1)
        assert page['conversations']==[] and page['conversation_count']==1 and page['trace_count']==summary['trace_count'] and not page['conversation_has_more']
        for query in ('conversation_offset=10001','conversation_limit=0','conversation_limit=101'):
            reject('observability/summary?'+query)
        assert Studio(base).trace_summary(project_id='absent')['trace_count']==0
        reject('observability/summary?since_ms=2&until_ms=1')
        assert len(requests)==before_summary
        # Native time-series route: immutable external telemetry, never inference.
        for index,started in enumerate((10,19,30)):
            currency='EUR' if index==1 else 'USD'
            usage=dict(input_tokens=7,output_tokens=2,first_text_ms=0,cost=index+1,cost_currency=currency)
            Studio(base).ingest_trace('default','series-fixture-'+str(index),started,[
                dict(parent_id=None,kind='agent',name='series-fixture',status='failed' if index==1 else 'completed',started_ms=started,duration_ms=5),
                dict(parent_id=0,kind='model',name='series-model',status='completed',started_ms=started,duration_ms=5,usage=usage),
                dict(parent_id=0,kind='tool',name='guardrail.input.block.fail.'+'a'*64,status='failed',started_ms=started,duration_ms=0)])
        series=Studio(base).trace_time_series(project_id='default',since_ms=10,until_ms=30,bucket_ms=10)
        assert series['kind']=='trace_time_series' and series['provider_calls']==0 and series['bucket_count']==3
        first,empty,last=series['buckets']
        assert first['trace_count']==2 and first['model_usage']['input_tokens']==14
        assert first['model_usage']['reported_cost_by_currency']=={'USD':1.0,'EUR':2.0}
        assert first['model_usage']['first_text']['known_calls']==2 and first['model_usage']['first_text']['p50_ms']==0
        assert first['guardrails']['blocked']==2 and empty['trace_count']==0 and empty['model_usage']['first_text']['p95_ms'] is None
        assert last['trace_count']==1 and last['start_ms']==30 and last['end_exclusive_ms']==31
        failed_series=Studio(base).trace_time_series(project_id='default',since_ms=10,until_ms=30,bucket_ms=10,status='failed')
        assert failed_series['status']=='failed' and failed_series['buckets'][0]['trace_count']==1 and failed_series['buckets'][0]['model_usage']['input_tokens']==7
        assert failed_series['buckets'][0]['model_usage']['reported_cost_by_currency']=={'EUR':2.0} and failed_series['buckets'][2]['trace_count']==0
        reject('observability/time-series?since_ms=10&until_ms=30&bucket_ms=10&status=INVALID')
        assert Studio(base).trace_time_series(project_id='absent',since_ms=10,until_ms=30,bucket_ms=10)['buckets'][0]['trace_count']==0
        for query in ('since_ms=30&until_ms=10&bucket_ms=10','since_ms=0&until_ms=500&bucket_ms=1','since_ms=0&until_ms=10&bucket_ms=0','since_ms=0&until_ms=10&bucket_ms=1&unknown=true'):
            reject('observability/time-series?'+query)
        assert len(requests)==before_summary
        print('PASS native trace time series route, boundary buckets, empty intervals, currencies, first text, guardrails and no inference')
        (workspace/'recovery-loop.txt').write_text('loop recovered')
        act(loop_session,'resume');recovered_loop=wait(loop_session,lambda s:s['status']=='paused' and 'Цель не завершена' in (s.get('notice') or ''))
        assert recovered_loop['messages'][-1]['content']=='Recovered after explicit resume.' and not (workspace/'blocked-loop.txt').exists()
        assert any(message['role']=='tool' and 'loop recovered' in message['content'] for message in recovered_loop['messages'])
        assert trace_rows(loop_session)[0]['status']=='goal_incomplete'
        print('PASS repeated identical tool failures pause Goal, preserve protocol, cancel writes and resume tools after repair while rejecting completion without a plan')
        other=create();act(other,'send','READ_SECOND');s=wait(other,lambda s:s['status']=='idle' and len(s['messages'])>=4);assert 'second-root-context' in next(m['content'] for m in s['messages'] if m['role']=='tool')
        for count in [2,3]:
            try:api(f'sessions/{auto}/branch',dict(message_count=count));raise AssertionError('split tool exchange accepted')
            except urllib.error.HTTPError as e:assert e.code==400
        tool_branch=api(f'sessions/{auto}/branch',dict(message_count=4))['id']
        assert api('sessions/'+tool_branch)['messages']==api('sessions/'+auto)['messages']
        print('PASS Plan read-only, Auto writes, multi-root project context and tool-safe branches')
        edited=create(mode='auto',allow_writes=True);act(edited,'send','TRY_EDIT')
        edited_state=wait(edited,lambda s:s['status']=='idle' and len(s['messages'])>=4)
        assert (workspace/'result.txt').read_text()=='verified edit'
        result=json.loads(next(m['content'] for m in edited_state['messages'] if m['role']=='tool'))
        assert '-verified write' in result['diff'] and '+verified edit' in result['diff']
        # The returned patch reconstructs the actual file exactly, including missing final newline.
        patch_root=root/'patch-check';patch_root.mkdir();(patch_root/'result.txt').write_text('verified write')
        subprocess.run(['git','apply','-'],input=result['diff'].encode(),cwd=patch_root,check=True)
        assert (patch_root/'result.txt').read_bytes()==(workspace/'result.txt').read_bytes()
        readonly=create(mode='auto',allow_writes=True);act(readonly,'send','TRY_EDIT READONLY')
        denied=wait(readonly,lambda s:s['status']=='idle' and len(s['messages'])>=4)
        assert 'read-only' in next(m['content'] for m in denied['messages'] if m['role']=='tool')
        print('PASS precise edits, applicable actual diff and read-only root enforcement')
        searched=create();act(searched,'send','TRY_AGENTGREP')
        searched_state=wait(searched,lambda s:s['status']=='idle' and len(s['messages'])>=4)
        found=json.loads(next(m['content'] for m in searched_state['messages'] if m['role']=='tool'))
        assert any(m['path']=='result.txt' for m in found['matches'])
        bg=create(mode='auto');act(bg,'send','TRY_BACKGROUND')
        bg_state=wait(bg,lambda s:s['status']=='idle' and len(s['messages'])>=4)
        bg_id=json.loads(next(m['content'] for m in bg_state['messages'] if m['role']=='tool'))['id']
        result=Studio(base).wait_background(bg,[bg_id],wait_seconds=3)
        assert result['tasks'][0]['status']=='completed' and result['tasks'][0]['stdout']=='background-done'
        assert result['tasks'][0]['name']=='Fixture background'
        background_totals=Studio(base).background_tasks(bg)['summary']
        assert background_totals['status_counts']['completed']==1 and background_totals['unknown_duration_count']==0
        assert background_totals['known_duration_ms']==result['tasks'][0]['duration_ms']
        assert background_totals['duration_semantics']=='sum_task_elapsed'

        background_metadata=Studio(base).export_background(bg)
        assert background_metadata['kind']=='background_export' and not background_metadata['outputs_included']
        assert all('stdout' not in task and 'stderr' not in task and 'error' not in task for task in background_metadata['tasks'])
        background_answers=Studio(base).export_background(bg,include_outputs=True)
        assert any(task['stdout']=='background-done' for task in background_answers['tasks'])

        temporary=Studio(base).start_background(bg,'kill -TERM $$',name='Temporary receipt')
        signalled=Studio(base).wait_background(bg,[temporary['id']],wait_seconds=5)['tasks'][0]
        assert signalled['status']=='failed' and signalled['exit_code'] is None and signalled['signal']==15
        assert Studio(base).cleanup_background(bg,temporary['id'])['removed']==1
        assert Studio(base).background_output(bg,bg_id)['stdout']=='background-done'
        assert temporary['id'] not in [task['id'] for task in Studio(base).background_tasks(bg)['tasks']]

        assert Studio(base).background_output(bg,bg_id)==result['tasks'][0]
        assert Studio(base).background_tasks(bg)['persistent'] is True
        summary=api('sessions/'+bg)['background']['tasks'][0]
        assert 'stdout' not in summary and summary['stdout_bytes']==15
        try:Studio(base).background_output(searched,bg_id);raise AssertionError('foreign task visible')
        except EvaluationError as e:assert e.reason=='http_400'
        try:Studio(base).start_background(searched,'printf forbidden');raise AssertionError('Chat started command')
        except EvaluationError as e:assert e.reason=='http_403'
        cancelled=Studio(base).start_background(bg,'sleep 30',timeout=30)['id']
        assert Studio(base).cancel_background(bg,cancelled)['cancel_requested'] is True
        assert Studio(base).wait_background(bg,[cancelled],wait_seconds=3)['tasks'][0]['status']=='cancelled'
        print('PASS source search tool roundtrip and background execution, output, wait, cancel, permissions and ownership')
        wake_session=create(mode='auto');act(wake_session,'send','wake setup');wait(wake_session,lambda s:s['status']=='idle' and len(s['messages'])==2)
        wake_task=Studio(base).start_background(wake_session,'sleep 0.05',follow_up='Continue after the command. Inspect the result.')
        wake_state=wait(wake_session,lambda s:s['status']=='idle' and len(s['messages'])==4)
        assert wake_task['id'] in wake_state['messages'][2]['content'] and 'Continue after the command' in wake_state['messages'][2]['content']
        stopped_wake=create(mode='auto');act(stopped_wake,'send','wake stop setup');wait(stopped_wake,lambda s:s['status']=='idle' and len(s['messages'])==2)
        act(stopped_wake,'stop');wait(stopped_wake,lambda s:s['status']=='paused')
        before_stopped_wake=len(requests)
        ignored_task=Studio(base).start_background(stopped_wake,'sleep 0.05',follow_up='Do not override Stop.')
        Studio(base).wait_background(stopped_wake,[ignored_task['id']],wait_seconds=3);time.sleep(.2)
        assert api('sessions/'+stopped_wake)['status']=='paused' and len(api('sessions/'+stopped_wake)['messages'])==2 and len(requests)==before_stopped_wake
        stale_wake=create(mode='auto');act(stale_wake,'send','stale setup');wait(stale_wake,lambda s:s['status']=='idle' and len(s['messages'])==2)
        stale_release=pathlib.Path(tmp)/'release-stale-background'
        stale_task=Studio(base).start_background(stale_wake,'while [ ! -f '+__import__('shlex').quote(str(stale_release))+' ]; do sleep 0.01; done',timeout=10,follow_up='STALE_FOLLOW_UP')
        act(stale_wake,'stop');wait(stale_wake,lambda s:s['status']=='paused')
        act(stale_wake,'send','New task after Stop');wait(stale_wake,lambda s:s['status']=='idle' and len(s['messages'])==4)
        stale_release.write_text('release')
        Studio(base).wait_background(stale_wake,[stale_task['id']],wait_seconds=3);time.sleep(.2)
        stale_state=api('sessions/'+stale_wake)
        assert len(stale_state['messages'])==4 and not stale_state['queue'] and 'STALE_FOLLOW_UP' not in json.dumps(stale_state['messages'])
        print('PASS explicit background follow-up continues Auto once and never overrides stopped session',flush=True)

        traces=api('observability/traces?session_id='+bg)['traces']
        assert traces[0]['status']=='completed'
        assert [span['kind'] for span in traces[0]['spans']]==['turn','model','tool','model']
        assert traces[0]['spans'][2]['name']=='background'
        assert all(span['parent_id']==0 for span in traces[0]['spans'][1:])
        assert 'background-done' not in json.dumps(traces)
        print('PASS tool trace hierarchy and no command/output capture')
        vision=create();act(vision,'send','image',images=[dict(name='tiny.png',mime='image/png',data='iVBORw0KGgo=')]);s=wait(vision,lambda s:s['status']=='idle' and len(s['messages'])==2);assert requests[-1]['messages'][-1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
        print('PASS image payload translation and history')
        # Token exhaustion is a resumable pause and retains queued messages.
        limited=create();act(limited,'send','TOKEN_LIMIT_TEXT');wait(limited,lambda s:s['status']=='running')
        act(limited,'send','AFTER-LIMIT')
        s=wait(limited,lambda s:s['status']=='paused');assert s['error'] is None and s['notice']
        assert s['messages'][-1]['content']=='partial text' and s['messages'][-1]['truncated']
        assert s['usage']['completion_tokens']==4096 and s['queue'][0]['text']=='AFTER-LIMIT'
        act(limited,'resume',settings=dict(settings,max_output_tokens=16384))
        s=wait(limited,lambda s:s['status']=='idle' and not s['queue'] and 'AFTER-LIMIT' in s['messages'][-1]['content'])
        resumed=[r for r in requests if 'The previous response reached' in str(r['messages'][-1].get('content',''))]
        assert resumed and resumed[-1]['max_tokens']==16384
        assert not any('truncated' in m or 'incomplete_tool_calls' in m for m in resumed[-1]['messages'])
        partial_tool=create(mode='auto',allow_writes=True);act(partial_tool,'send','TOKEN_LIMIT_TOOL')
        s=wait(partial_tool,lambda s:s['status']=='paused');assert s['messages'][-1]['incomplete_tool_calls'] and not s['messages'][-1].get('tool_calls')
        assert not any(m['role']=='tool' for m in s['messages'])
        print('PASS token-limit pause, partial output/usage, safe tools, configurable resume and retained queue')
        api('providers/deepseek/key',dict(key='dummy-validation-only',persist=False))
        for model in ['deepseek-flash','deepseek-v4-flash']:
            flash=create(provider='deepseek',model=model,max_output_tokens=393216,compact_threshold=550000)
            assert api('sessions/'+flash)['settings']['max_output_tokens']==393216
        high=create(provider='deepseek',model='deepseek-v4-pro',max_output_tokens=393216,compact_threshold=550000)
        assert api('sessions/'+high)['settings']['max_output_tokens']==393216
        assert api('sessions/'+high)['context_stats']['context_window']==1000000
        try:
            create(max_output_tokens=393216)
            raise AssertionError('Unknown model accepted DeepSeek-only limit')
        except urllib.error.HTTPError as e: assert e.code==400
        print('PASS DeepSeek Pro output maximum and model-specific validation')
        verbose=create(verbosity='maximum');act(verbose,'send','DETAIL_TEST',settings=dict(settings,verbosity='maximum'))
        detailed=wait(verbose,lambda s:s['status']=='idle' and len(s['messages'])==2)
        assert detailed['settings']['verbosity']=='maximum'
        assert 'Response detail: maximum.' in requests[-1]['messages'][0]['content']
        print('PASS verbosity stored and included in provider instructions')
        thinking=create();act(thinking,'send','STREAM_ANALYSIS')
        streamed=wait(thinking,lambda s:s['status']=='running' and s['messages'][-1].get('reasoning_content'))
        assert streamed['messages'][-1]['content']==''
        finished=wait(thinking,lambda s:s['status']=='idle')
        assert finished['messages'][-1]['reasoning_content']=='Visible analysis <script>literal</script>'
        assert finished['messages'][-1]['content']=='Finished answer'
        print('PASS reasoning streams before answer and remains in history')
        compact_guard=create(auto_compact=False)
        for turn in range(4):
            act(compact_guard,'send','guarded history '+('context '*1000))
            wait(compact_guard,lambda s:s['status']=='idle' and len(s['messages'])==(turn+1)*2)
        compact_original=api('sessions/'+compact_guard)['messages']
        compact_input_policy=Studio(base).save_guardrail_policy([dict(id='transcript',kind='forbidden_substrings',value=['user:'])])['policy']['policy_sha256']
        compact_output_policy=Studio(base).save_guardrail_policy([dict(id='summary',kind='forbidden_substrings',value=['COMPACTED:'])])['policy']['policy_sha256']
        compact_guard_settings=dict(settings,auto_compact=False,guardrails=dict(input_policy_sha256=compact_input_policy,output_policy_sha256=policy_hash,action='block'))
        before_compact_guard=len(requests)
        act(compact_guard,'compact',settings=compact_guard_settings)
        compact_rejected=wait(compact_guard,lambda s:s['status']!='running')
        assert len(requests)==before_compact_guard and compact_rejected['compaction'] is None and compact_rejected['messages']==compact_original
        compact_guard_settings['guardrails']=dict(input_policy_sha256=policy_hash,output_policy_sha256=compact_output_policy,action='block')
        act(compact_guard,'compact',settings=compact_guard_settings)
        compact_rejected=wait(compact_guard,lambda s:s['status']!='running')
        assert len(requests)==before_compact_guard+1 and compact_rejected['compaction'] is None and compact_rejected['messages']==compact_original
        compact_guard_settings['guardrails']['action']='observe'
        act(compact_guard,'compact',settings=compact_guard_settings)
        compact_observed=wait(compact_guard,lambda s:s['status']=='idle' and s.get('compaction'))
        assert compact_observed['messages']==compact_original and compact_observed['compaction']['summary'].startswith('COMPACTED:')
        print('PASS compaction policies input no inference, blocked summary rollback and observed summary without history changes',flush=True)
        compacted=create(auto_compact=False)
        for i in range(4):
            act(compacted,'send',f'long turn {i} '+('context '*1000))
            wait(compacted,lambda s:s['status']=='idle' and len(s['messages'])==(i+1)*2)
        original=api('sessions/'+compacted)['messages']
        act(compacted,'compact')
        summary=wait(compacted,lambda s:s['status']=='idle' and s.get('compaction'))
        assert summary['messages']==original and summary['compaction']['through']==4
        compact_trace=next(t for t in trace_rows(compacted) if t['spans'][0]['kind']=='compaction')
        assert [span['kind'] for span in compact_trace['spans']]==['compaction','model']
        assert compact_trace['status']=='completed' and compact_trace['spans'][1]['parent_id']==0
        assert_trace_tree(compact_trace)
        assert 'long turn 0' not in json.dumps(compact_trace)

        stats=summary['context_stats']
        assert stats['compacted_messages']==4 and stats['messages']==8
        assert stats['estimated_history_tokens'] < stats['original_history_tokens']
        assert stats['remaining_before_compact']==max(0,stats['compact_threshold']-stats['estimated_history_tokens'])
        act(compacted,'send','AFTER-COMPACT')
        wait(compacted,lambda s:s['status']=='idle' and len(s['messages'])==10)
        assert 'COMPACTED:' in requests[-1]['messages'][1]['content']
        assert not any('long turn 0' in m.get('content','') for m in requests[-1]['messages'])
        act(compacted,'send','TRY_CONVERSATION_SEARCH')
        recalled=wait(compacted,lambda s:s['status']=='idle' and len(s['messages'])==14)
        result=json.loads(next(m['content'] for m in reversed(recalled['messages']) if m['role']=='tool'))
        assert any(m['compacted'] and m['message_index']==0 for m in result['matches'])
        print('PASS agent retrieves original conversation messages hidden by compaction')
        retry=create(auto_compact=False)
        for i in range(3):
            act(retry,'send','RETRY_COMPACT '+str(i)+(' context'*100))
            wait(retry,lambda s:s['status']=='idle' and len(s['messages'])==(i+1)*2)
        original_retry=api('sessions/'+retry)['messages']; first_request=len(requests)
        act(retry,'compact')
        progress=wait(retry,lambda s:'Сжатие контекста: часть 1' in (s.get('notice') or ''))
        assert progress['messages']==original_retry and not progress.get('compaction')
        retried=wait(retry,lambda s:s['status']=='idle' and s.get('compaction'))
        attempts=requests[first_request:]
        assert [r['max_tokens'] for r in attempts]==[16384,32768]
        assert attempts[0]['messages']==attempts[1]['messages']
        assert retried['messages']==original_retry

        retry_trace=next(t for t in trace_rows(retry) if t['spans'][0]['kind']=='compaction')
        assert [span['status'] for span in retry_trace['spans']]==['completed','token_limit','completed']
        assert_trace_tree(retry_trace)
        print('PASS compaction retries identical input with larger budget and preserves history')
        automatic=create(auto_compact=True,compact_threshold=4096)
        for i in range(4):
            act(automatic,'send',f'auto turn {i} '+('context '*1000))
            wait(automatic,lambda s:s['status']=='idle' and len(s['messages'])==(i+1)*2)
        assert api('sessions/'+automatic)['compaction']['through']>=2
        auto_traces=trace_rows(automatic)
        auto_trace=next(t for t in auto_traces if any(span['kind']=='compaction' for span in t['spans']))
        phase=next(span for span in auto_trace['spans'] if span['kind']=='compaction')
        assert phase['name']=='automatic' and phase['parent_id']==0
        assert any(span['kind']=='model' and span['parent_id']==phase['id'] for span in auto_trace['spans'])
        assert_trace_tree(auto_trace)

        failed=create(auto_compact=False)
        for i in range(3):
            act(failed,'send','FAIL_COMPACT '+str(i))
            wait(failed,lambda s:s['status']=='idle' and len(s['messages'])==(i+1)*2)
        before=api('sessions/'+failed)['messages'];act(failed,'compact')
        unchanged=wait(failed,lambda s:s['status']=='error')
        assert unchanged['compaction'] is None and unchanged['messages']==before

        failed_trace=next(t for t in trace_rows(failed) if t['spans'][0]['kind']=='compaction')
        assert failed_trace['status']=='failed' and all(span['status']=='token_limit' for span in failed_trace['spans'][1:])
        assert_trace_tree(failed_trace)
        compact_stop=create(auto_compact=False)
        for turn in range(3):
            act(compact_stop,'send','RETRY_COMPACT stop '+str(turn)+' context'*100)
            wait(compact_stop,lambda state:state['status']=='idle' and len(state['messages'])==(turn+1)*2)
        act(compact_stop,'compact');wait(compact_stop,lambda state:'Сжатие контекста: часть 1' in (state.get('notice') or ''))
        act(compact_stop,'stop');wait(compact_stop,lambda state:state['status']=='paused')
        stopped_trace=next(t for t in trace_rows(compact_stop) if t['spans'][0]['kind']=='compaction')
        assert stopped_trace['status']=='interrupted';assert_trace_tree(stopped_trace)
        print('PASS complete manual/auto compaction trace trees, token-limit retries, failures, cancellation and content privacy')
        print('PASS manual/auto compaction, reduced API context, original history and failed-summary rollback')
        # Full-text search, reversible history management and inert imports.
        found=api('sessions?q=AFTER-COMPACT&project=default')
        assert compacted in [x['id'] for x in found] and found[0]['match_preview']
        assert api('sessions?q=AFTER-COMPACT&project=missing')==[]
        act(compacted,'rename','Renamed conversation');assert api('sessions/'+compacted)['title']=='Renamed conversation'
        original=api('sessions/'+compacted)['messages']
        act(compacted,'move','archived')
        assert compacted not in [x['id'] for x in api('sessions')]
        assert compacted in [x['id'] for x in api('sessions?folder=archived')]
        try:act(compacted,'send','blocked');raise AssertionError('archived conversation ran')
        except urllib.error.HTTPError as e:assert e.code==409
        act(compacted,'move','trash');assert api('sessions/'+compacted)['messages']==original
        act(compacted,'move','active');assert api('sessions/'+compacted)['messages']==original
        exported=api('sessions/'+edited);exported['settings']['allow_writes']=True;exported['settings']['mode']='auto'
        exported['queue']=[dict(text='MUST NOT RUN',images=[],settings=settings)]
        exported['settings']['guardrails']=dict(input_policy_sha256=policy_hash,output_policy_sha256=policy_hash,action='block')
        before_import_calls=len(requests)
        imported=api('sessions/import',dict(session=exported,project_id='default'))['id']
        imported_state=api('sessions/'+imported)
        assert imported!=edited and imported_state['messages']==exported['messages']
        assert imported_state['settings']['guardrails'] is None and len(requests)==before_import_calls
        assert not imported_state['queue'] and not imported_state['settings']['allow_writes'] and imported_state['settings']['mode']=='chat'
        invalid=dict(exported,messages=[dict(role='tool',content='orphan',tool_call_id='missing')])
        try:api('sessions/import',dict(session=invalid,project_id='default'));raise AssertionError('orphan tool import accepted')
        except urllib.error.HTTPError as e:assert e.code==400
        act(imported,'move','trash')
        print('PASS full-text search, rename, archive/trash/restore, isolated inert import and invalid exchange rejection')
        # Swarm: a wave of independent members, one merged MASTER, no writes.
        two_members=[dict(label='scout',role='map the project',provider='local',model='mock'),
                     dict(label='risks',role='find regressions',provider='local',model='mock')]
        swarm=create(mode='swarm',swarm=dict(members=two_members,max_steps_per_member=1,report_bytes=4000))
        started=len(requests)
        act(swarm,'send','SWARM-BRIEF')
        state=wait(swarm,lambda s:s['status']=='idle' and s['messages'][-1].get('swarm'))
        last=state['messages'][-1]
        assert [r['label'] for r in last['swarm']]==['scout','risks']
        assert all(r['status']=='done' and r['provider']=='local' and r['model']=='mock' for r in last['swarm'])
        assert [r['content'] for r in last['swarm']]==['REPORT from scout','REPORT from risks']
        assert last['content']=='MASTER merged answer'
        wave=[r for r in requests[started:] if 'Ты — участник swarm' in str(r['messages'][-1]['content'])]
        merged=[r for r in requests[started:] if 'Собери из них один итоговый ответ' in str(r['messages'][-1]['content'])]
        assert len(requests)-started==3 and len(wave)==2 and len(merged)==1
        member_tools=[t['function']['name'] for t in wave[0]['tools']]
        assert sorted(member_tools)==['agentgrep','list_files','read_file'] and 'write_file' not in member_tools
        assert all('swarm' not in message for message in merged[0]['messages'])
        assert state['usage']['completion_tokens']==20 and state['usage']['prompt_tokens']==17
        print('PASS swarm wave runs in parallel, merges to MASTER, members stay read-only and usage is summed')
        stepping=create(mode='swarm',swarm=dict(members=two_members,max_steps_per_member=2,report_bytes=4000))
        act(stepping,'send','SWARM-STEP brief')
        state=wait(stepping,lambda s:s['status']=='idle' and s['messages'][-1].get('swarm'))
        stepped={r['label']:r for r in state['messages'][-1]['swarm']}
        assert stepped['scout']['status']=='done' and stepped['scout']['step']==2, stepped['scout']
        assert stepped['risks']['step']==1, stepped['risks']
        # The word stays the phase; the step it used to carry is a field, so a
        # client never has to prefix-match a status again.
        assert all(r['status'] in ('queued','running','done','cancelled','error') for r in stepped.values())

        step_trace=trace_rows(stepping)[0];assert_trace_tree(step_trace)
        members={span['id']:span for span in step_trace['spans'] if span['kind']=='swarm_member'}
        assert len(members)==2
        read_span=next(span for span in step_trace['spans'] if span['kind']=='tool')
        assert read_span['name']=='read_file' and members[read_span['parent_id']]['name'].startswith('scout')
        models=[span for span in step_trace['spans'] if span['kind']=='model'];assert len(models)==4
        assert sum(span['usage'].get('completion_tokens',0) for span in models)==state['usage']['completion_tokens']
        assert all(not span['usage'] for span in step_trace['spans'] if span['kind']!='model')
        assert 'second-root-context' not in json.dumps(step_trace) and 'SWARM-STEP brief' not in json.dumps(step_trace)
        print('PASS swarm member reports its tool-loop step as a field beside the status word')
        limited=create(mode='swarm',swarm=dict(members=two_members,max_steps_per_member=1))
        act(limited,'send','SWARM-STEP limited')
        limited_state=wait(limited,lambda state:state['status']=='idle' and state['messages'][-1].get('swarm'))
        limited_trace=trace_rows(limited)[0];assert_trace_tree(limited_trace)
        assert next(span for span in limited_trace['spans'] if span['kind']=='swarm_member' and span['name'].startswith('scout'))['status']=='step_limit'
        assert not any(span['kind']=='tool' for span in limited_trace['spans'])
        assert next(report for report in limited_state['messages'][-1]['swarm'] if report['label']=='scout')['status']=='error'

        for broken in [dict(members=[dict(label='solo',role='',provider='local',model='mock')]),
                       dict(members=[dict(label='a',role='',provider='ghost',model='mock'),dict(label='b',role='',provider='local',model='mock')]),
                       dict(members=[dict(label='a',role='',provider='local',model='mock'),dict(label='A',role='',provider='local',model='mock')])]:
            try:
                create(mode='swarm',swarm=broken)
                raise AssertionError('invalid swarm accepted')
            except urllib.error.HTTPError as e:assert e.code==400
        print('PASS swarm validation rejects one member, unknown provider and duplicate names')
        partial=create(mode='swarm',swarm=dict(members=two_members,max_steps_per_member=1))
        act(partial,'send','SWARM-FAIL brief')
        state=wait(partial,lambda s:s['status']=='idle' and s['messages'][-1].get('swarm'))
        reports={r['label']:r for r in state['messages'][-1]['swarm']}
        assert reports['risks']['status']=='error' and reports['risks']['error']
        assert reports['scout']['status']=='done' and reports['scout']['content']=='REPORT from scout'
        assert 'не ответил' in state['notice'] and 'MASTER merged answer' in state['messages'][-1]['content']

        partial_trace=trace_rows(partial)[0];assert_trace_tree(partial_trace)
        failed_member=next(span for span in partial_trace['spans'] if span['kind']=='swarm_member' and span['name'].startswith('risks'))
        assert failed_member['status']=='failed' and any(span['status']=='failed' and span['parent_id']==failed_member['id'] for span in partial_trace['spans'])
        print('PASS swarm names a failed member instead of inventing its report')
        def merges():return len([r for r in requests if 'Собери из них один итоговый ответ' in str(r['messages'][-1]['content'])])
        retried=create(mode='swarm',swarm=dict(members=two_members,max_steps_per_member=1))
        act(retried,'send','SWARM-FAIL retry brief')
        state=wait(retried,lambda s:s['status']=='idle' and s['messages'][-1].get('swarm') and all(r['status']!='queued' for r in s['messages'][-1]['swarm']))
        assert {r['label']:r['status'] for r in state['messages'][-1]['swarm']}=={'scout':'done','risks':'error'}
        before=merges()
        previous_retries=len([r for r in requests if 'Это повтор' in str(r['messages'][-1]['content'])])
        act(retried,'retry_member','risks')
        state=wait(retried,lambda s:s['status']=='idle' and all(r['status']=='done' for r in s['messages'][-1]['swarm']))
        reports={r['label']:r for r in state['messages'][-1]['swarm']}
        assert reports['risks']['content']=='REPORT from risks' and not reports['risks'].get('error')
        assert reports['scout']['content']=='REPORT from scout',repr(reports)
        assert len([r for r in requests if 'Это повтор' in str(r['messages'][-1]['content'])])==previous_retries+1
        assert merges()==before+1 and not state['notice']
        assert state['usage']['prompt_tokens']==12 and state['usage']['completion_tokens']==14
        for bad in '','  ':
            try:act(retried,'retry_member',bad);raise AssertionError('nameless retry accepted')
            except urllib.error.HTTPError as e:assert e.code==400
        act(retried,'retry_member','nobody')
        state=wait(retried,lambda s:s['status']=='error')
        assert 'nobody' in state['error'] and state['messages'][-1]['content']=='MASTER merged answer'
        assert all(r['status']=='done' for r in state['messages'][-1]['swarm'])
        plain=create()
        try:act(plain,'retry_member','risks');raise AssertionError('retry accepted outside swarm')
        except urllib.error.HTTPError as e:assert e.code==429
        print('PASS swarm retry re-runs one member, rebuilds the MASTER and bills only the retry')
        previous_critics=len([r for r in requests if 'Ты — критик swarm-результата' in str(r['messages'][-1]['content'])])
        critic=create(mode='swarm',swarm=dict(members=two_members,max_steps_per_member=1,critic=True))
        act(critic,'send','SWARM-CRITIC')
        state=wait(critic,lambda s:s['status']=='idle' and s['messages'][-1].get('swarm'))
        assert state['messages'][-1]['content']=='MASTER merged answer (critic passed)'
        assert len([r for r in requests if 'Ты — критик swarm-результата' in str(r['messages'][-1]['content'])])==previous_critics+1
        assert state['usage']['completion_tokens']==6+6+8+10

        critic_trace=trace_rows(critic)[0];assert_trace_tree(critic_trace)
        phases=[span for span in critic_trace['spans'] if span['kind'] in ['synthesis','critic']]
        assert [phase['kind'] for phase in phases]==['synthesis','critic']
        assert all(any(span['kind']=='model' and span['parent_id']==phase['id'] for span in critic_trace['spans']) for phase in phases)
        assert sum(span['usage'].get('completion_tokens',0) for span in critic_trace['spans'] if span['kind']=='model')==state['usage']['completion_tokens']
        print('PASS swarm critic pass replaces the draft only after it succeeds')
        stopping=create(mode='swarm',swarm=dict(members=[dict(label='slow',role='collect',provider='local',model='mock'),dict(label='risks',role='regressions',provider='local',model='mock')],max_steps_per_member=1))
        act(stopping,'send','SWARM-STOP')
        wait(stopping,lambda s:s['status']=='running' and 'REPORT from slow' in json.dumps(s['messages'][-1].get('swarm') or []))
        act(stopping,'stop')
        stopped=wait(stopping,lambda s:s['status']=='paused')
        statuses=[r['status'] for r in stopped['messages'][-1]['swarm']]
        assert not any(status=='running' or status=='queued' for status in statuses), statuses
        assert 'cancelled' in statuses and stopped['messages'][-1]['content']==''

        stop_trace=trace_rows(stopping)[0];assert stop_trace['status']=='interrupted';assert_trace_tree(stop_trace)
        assert any(span['kind']=='model' and span['status']=='interrupted' for span in stop_trace['spans'])
        print('PASS complete swarm member/model/tool/synthesis/critic traces, reported usage, errors, cancellation and metadata privacy')
        print('PASS stop cancels the whole wave and marks unfinished reports honestly')
        swarm_export=api('sessions/'+swarm)
        swarm_import=api('sessions/import',dict(session=swarm_export,project_id='default'))['id']
        imported_swarm=api('sessions/'+swarm_import)
        assert imported_swarm['messages']==swarm_export['messages'] and imported_swarm['settings']['mode']=='chat'
        print('PASS swarm reports survive export/import without re-running the wave')
        # Persisted projects and conversations survive restart.
        # Deterministic journal fixture for the crash window between variants:
        # completed first run, reserved-but-unsent second run, unreserved third.
        import uuid
        pending_matrix_journal=json.loads((data/'evaluation'/'matrix-jobs'/(completed_job['id']+'.json')).read_text())
        pending_matrix_journal.update(id=uuid.uuid4().hex,matrix_id=uuid.uuid4().hex,status='running',error=None)
        unsent_reservation=uuid.uuid4().hex
        pending_matrix_journal['run_ids']=[completed_job['variants'][0]['run_id'],unsent_reservation,None]
        pending_matrix_journal['run_attempted']=[True,False,False]
        (data/'evaluation'/'matrix-jobs'/(pending_matrix_journal['id']+'.json')).write_text(json.dumps(pending_matrix_journal))
        process.terminate();process.wait(timeout=5)
        interrupted_request=dict(quality_job['plan']['request'],rubric='Interrupted quality fixture')
        interrupted_quality_id=hashlib.sha256(json.dumps(interrupted_request,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        interrupted_quality_dir=data/'evaluation/quality-jobs'/interrupted_quality_id;interrupted_quality_dir.mkdir()
        (interrupted_quality_dir/'plan.json').write_text(json.dumps(dict(kind='online_quality_job',schema_version=1,id=interrupted_quality_id,request=interrupted_request)))
        (interrupted_quality_dir/'claim.json').write_text(json.dumps(dict(kind='online_quality_job_claim',schema_version=1,id=interrupted_quality_id)))
        # Mutate only isolated test-owned histories while the server is stopped.
        for label,source_session,retained in extraction_status_fixtures:
            source_path=data/(source_session+'.json')
            if label=='unavailable':source_path.rename(root/('unavailable-history-'+source_session+'.json'))
            else:
                source_history=json.loads(source_path.read_text());source_history['messages'][0]['content']='REWRITTEN_SOURCE_STATUS';source_path.write_text(json.dumps(source_history))
        quality_restart_calls=len(requests)
        process=start()
        assert Studio(base).online_quality_job(quality_job['id'])==quality_job
        interrupted_quality=Studio(base).online_quality_job(interrupted_quality_id)
        assert interrupted_quality['status']=='interrupted' and interrupted_quality['result']['judge_id'] is None
        quality_catalog=Studio(base).online_quality_jobs('default',limit=1);quality_next=Studio(base).online_quality_jobs('default',offset=1,limit=1)
        assert quality_catalog['total']==3 and quality_catalog['has_more'] and quality_next['has_more']
        assert {row['status'] for row in Studio(base).online_quality_jobs('default')['jobs']}=={'completed','interrupted'}
        assert all(page['provider_calls']==0 and page['automatic_execution'] is False for page in [quality_catalog,quality_next])
        assert Studio(base).online_quality_jobs('global')['total']==0

        assert len(requests)==quality_restart_calls
        print('PASS background quality job deduplicates provider execution, retains linked verdict and recovers claimed requests as interrupted without replay',flush=True)
        for label,source_session,retained in extraction_status_fixtures:
            calls=len(requests);status_notes=api('memory/notes?project_id=default');checked=Studio(base).memory_extraction_source_status(retained['id'])
            assert checked['status']==label and checked['provider_calls']==0 and checked['notes_modified'] is False and len(requests)==calls
            assert api('memory/notes?project_id=default')==status_notes and Studio(base).memory_proposal(retained['id'])==retained
            assert checked['current_source_sha256']!=retained['source_sha256']
        print('PASS HTTP extraction prefix remains current after append, detects rewritten or unavailable history after restart without inference or receipt mutation',flush=True)
        assert 'BLOCKED_STREAM_TEXT' not in json.dumps(api('sessions/'+cancelled_guard));assert trace_rows(cancelled_guard)[0]['status']=='interrupted';assert all(api('observability/traces/'+trace_id)['spans'][1]['usage']['guardrail_receipt']['rules'] for trace_id in external_checks);assert any(span['usage'].get('guardrail_receipt') for span in trace_rows(streaming_guard)[0]['spans']);assert api('sessions/'+streaming_guard)['settings']['guardrails']['output_policy_sha256']==output_hash;assert 'BLOCKED_STREAM_TEXT' not in json.dumps(api('sessions/'+streaming_guard));assert Studio(base).guardrail_policy(policy_hash)==policy_receipt;assert api('sessions/'+id)['messages'];assert len(api('config')['projects'][0]['roots'])==2
        assert api('sessions/'+vision)['messages'][0]['images'][0]['name']=='tiny.png'
        assert api('sessions/'+branch)['parent']['session_id']==id
        assert api('sessions/'+id)['bookmarks']==marks and api('sessions/'+marked_copy)['bookmarks']==marks
        assert api('sessions/'+compacted)['compaction']==summary['compaction']
        assert api('sessions/'+imported)['folder']=='trash'
        assert api('sessions/'+swarm)['messages'][-1]['swarm'][0]['content']=='REPORT from scout'
        assert api('sessions/'+compacted)['title']=='Renamed conversation'
        assert api(f"memory/notes/{note['id']}/versions/2")==removed_note
        assert api('memory/proposals/'+proposal['id'])==proposal
        extraction_status_calls=len(requests);extraction_status_notes=api('memory/notes?project_id=default')
        extraction_status=Studio(base).memory_extraction_source_status(proposal['id'])
        assert extraction_status['kind']=='memory_extraction_source_status' and extraction_status['status']=='current' and extraction_status['semantics']=='extraction_prefix'
        assert extraction_status['source_sha256']==proposal['source_sha256']==extraction_status['current_source_sha256'] and extraction_status['message_count']==2
        assert extraction_status['provider_calls']==0 and extraction_status['notes_modified'] is False and len(requests)==extraction_status_calls
        assert 'content' not in extraction_status and api('memory/notes?project_id=default')==extraction_status_notes
        reject('memory/proposals/'+synthesis_proposal['id']+'/source-status')
        print('PASS extraction prefix status reports exact hash without source text, provider calls or note mutation',flush=True)

        assert Studio(base).memory_proposal(synthesis_proposal['id'])==synthesis_proposal and api(f"memory/notes/{synthesis_accepted['id']}/versions/1")==synthesis_accepted
        assert Studio(base).memory_consolidation_proposals('default',limit=1)==consolidation_catalog
        assert api(f"memory/notes/{accepted['id']}/versions/2")==updated
        assert api(f"memory/notes/{due_note['id']}/versions/1")==due_note
        assert api(f"memory/notes/{merged_note['id']}/versions/1")==merged_note and api(f"memory/notes/{merged_note['id']}/versions/2")==merged_revision
        assert api(f"memory/notes/{merge_first['id']}/versions/1")==merge_first
        assert Studio(base).memory_consolidation_source_status(merged_note['id'],1)==removed_sources
        restarted_expiry=Studio(base).memory_expiry('default',horizon_days=2)
        assert next(row for row in restarted_expiry['notes'] if row['id']==due_note['id'])['sha256']==due_note['sha256']
        print('PASS conversation, image, compaction and project persistence')
        assert api('observability/traces?session_id='+bg)['traces'][0]['spans'][2]['name']=='background'
        assert api(f'sessions/{bg}/background',dict(action='output',task_id=bg_id))['stdout']=='background-done'
        assert api('observability/traces/'+skipped_trace.receipt['id'])==skipped_record
        assert Studio(base).callback_evaluation_summary('default',**callback_summary_scope)==callback_summary
        assert api('observability/traces/'+online_trace.receipt['id'])==online_record
        assert Studio(base).online_evaluation_rule(online_hash)==online_snapshot and Studio(base).online_evaluation_rule(online_next['rule_sha256'])==online_next
        assert Studio(base).online_evaluation_rules('default',limit=1)==online_catalog and Studio(base).online_evaluation_rules('default',offset=1,limit=1)==online_catalog_next
        assert Studio(base).online_evaluation_binding('default','quality')==online_binding_status
        assert Studio(base).select_online_evaluations('default',external_receipt['id'])==online_selection
        assert Studio(base).online_evaluation_selection_archive('default',external_receipt['id'])==online_archive
        assert Studio(base).assess_online_trace('default',adapter_trace.receipt['id'])==health_assessment and Studio(base).assess_online_trace('default',responses_trace.receipt['id'])==unknown_assessment
        assert online_client.online_evaluation_selection_archive('default',online_auto_trace['id'])['selection']==auto_selection
        assert any(json.loads(path.read_text())==auto_job for path in auto_jobs.glob('*.json'))
        assert online_client.drain_online_evaluation_jobs(limit=20)==terminal_batch
        assert {path.name:path.read_bytes() for path in (data/'observability'/'online-rules'/'job-results').glob('*.json')}==online_result_files
        print('PASS online rule snapshots persist across restart without evaluator execution',flush=True)
        print('PASS trace persistence across Studio restart')
        assert api(feedback_endpoint)['version']==5 and api(feedback_endpoint+'?version=1')['annotations']==[saved_review]
        print('PASS feedback revisions survive Studio restart')
        restored_queue=Studio(base).review_queue(queue['id'])
        assert restored_queue==completed_queue and restored_queue['version']==13 and restored_queue['archived'] and restored_queue['completions']['0']['annotation_id']==saved_review['id']
        assert Studio(base).review_queue_history(queue['id'],limit=2)==queue_history
        print('PASS review queue assignment/completion evidence persists across restart without replay',flush=True)
        assert api(f"evaluation/datasets/{dataset['id']}/versions/1")['sha256']==dataset['sha256']
        assert api('evaluation/experiments/'+frozen_run)['items'][0]['scores']['exact_match']==1
        assert trace_rows('experiment-'+frozen_run)[0]['id']==frozen['trace_id']
        assert api(f"evaluation/prompts/{saved_prompt['id']}/versions?limit=1")['versions'][0]['sha256']==next_prompt['sha256']
        assert Studio(base).scored_outputs(offline['id'])==offline
        restored_scores=Studio(base).list_scored_outputs('default',dataset_id=f1_data['id'],limit=100)
        assert offline['id'] in [row['id'] for row in restored_scores['scores']]

        assert Studio(base).judgment(judged['id'])==judged
        assert Studio(base).judge_plan(judge_plan['id'])==judge_plan
        assert Studio(base).judge_run(native_judges)==native_completed
        assert any(row['id']==native_judges for row in Studio(base).judge_runs('default')['runs'])
        assert Studio(base).judge_run(cancelled_judges)==cancelled_result
        assert Studio(base).comparison(saved_experiment_comparison['id'])==saved_experiment_comparison
        assert Studio(base).experiment_matrix(saved_native_matrix['id'])==saved_native_matrix
        print('PASS dataset, experiment and reverified comparison persistence across restart')
        assert Studio(base).matrix_job(completed_job['id'])==completed_job
        recovered_pending_matrix=Studio(base).matrix_job(pending_matrix_journal['id'])
        assert recovered_pending_matrix['status']=='interrupted'
        before_unsent_resume=len(requests)
        Studio(base).resume_matrix_job(pending_matrix_journal['id'])
        continued_matrix=wait_matrix_job(pending_matrix_journal['id'])
        assert continued_matrix['status']=='completed' and len(requests)==before_unsent_resume+6,continued_matrix
        assert continued_matrix['variants'][0]['run_id']==completed_job['variants'][0]['run_id']
        assert continued_matrix['variants'][1]['run_id']==unsent_reservation
        assert not Studio(base).experiment_matrix(continued_matrix['matrix_id'])['passed']
        print('PASS explicit post-restart matrix continuation reuses completed run and unsent reservation without replay',flush=True)
        # SIGKILL bypasses all Drop guards: startup must close persisted orphans honestly.
        crash_data=api('evaluation/datasets',dict(project_id='default',name='Crash recovery',base_version=0,samples=[dict(id='finished',input='good',expected_output='answer: good'),dict(id='pending',input='SLOW_RECOVERY',expected_output='never')]))
        crash_run=experiment(crash_data)
        crash_matrix_variants=[dict(label=label,sha256=crash_data['sha256'],request=dict(dataset_id=crash_data['id'],dataset_version=1,settings=settings,metrics=['exact_match'],prompt_template='{{input}}',concurrency=2)) for label in ['interrupted first','never admitted']]
        crash_matrix_job=Studio(base).start_matrix_job('default',crash_matrix_variants)
        until=time.time()+5
        while time.time()<until:
            matrix_partial=Studio(base).matrix_job(crash_matrix_job['id'])
            matrix_owned_id=matrix_partial['variants'][0]['run_id']
            if matrix_owned_id:
                try:matrix_partial_run=Studio(base).run(matrix_owned_id)
                except EvaluationError as error:
                    assert error.reason=='http_400'
                else:
                    if matrix_partial_run['items'][0]['status']=='completed' and matrix_partial_run['items'][1]['status']=='pending':break
            time.sleep(.01)
        else:raise AssertionError('no partial server-owned matrix before crash')
        retry_crash_data=api('evaluation/datasets',dict(project_id='default',name='Matrix explicit retry',base_version=0,samples=[dict(id='sample',input='good',expected_output='answer: good')]))
        retry_crash_variants=[dict(label=label,sha256=retry_crash_data['sha256'],request=dict(dataset_id=retry_crash_data['id'],dataset_version=1,settings=dict(settings,model=model),metrics=['exact_match'],prompt_template=template)) for label,model,template in [('completed baseline','mock','{{input}}'),('interrupted variant','matrix-retry-crash','SLOW_MATRIX_RETRY {{input}}'),('unsent variant','mock','{{input}}')]]
        retry_source_job=Studio(base).start_matrix_job('default',retry_crash_variants)
        until=time.time()+5
        while time.time()<until:
            retry_partial=Studio(base).matrix_job(retry_source_job['id'])
            if retry_partial['variants'][1]['run_id'] and any(row.get('model')=='matrix-retry-crash' for row in requests):break
            time.sleep(.01)
        else:raise AssertionError('matrix retry fixture did not enter the second model call')
        before_active_retry=len(requests)
        try:Studio(base).retry_matrix_job(retry_source_job['id']);raise AssertionError('active job duplicated by retry')
        except EvaluationError as error:assert error.reason=='http_400'
        assert len(requests)==before_active_retry
        until=time.time()+5
        while time.time()<until:
            partial_run=api('evaluation/experiments/'+crash_run)
            if partial_run['items'][0]['status']=='completed' and partial_run['items'][1]['status']=='pending' and partial_run.get('trace_id'):break
            time.sleep(.01)
        else:raise AssertionError('no partial experiment receipt before crash')
        crash_chat=create();act(crash_chat,'send','SLOW_RECOVERY_CHAT')
        wait(crash_chat,lambda state:state['status']=='running' and state['messages'][-1]['content'].startswith('started'))
        chat_trace=trace_rows(crash_chat)[0];assert chat_trace['status']=='running'
        reject('observability/online-selections',dict(project_id='default',trace_id=chat_trace['id']))
        reject('observability/trace-exports',dict(project_id='default',trace_ids=[chat_trace['id']]))
        exp_trace=trace_rows('experiment-'+crash_run)[0];assert exp_trace['status']=='running'
        stable_trace_path=data/'observability'/'traces'/(frozen['trace_id']+'.json');stable_trace=stable_trace_path.read_bytes()
        interrupted_judges=Studio(base).start_judges(cancel_plan['id'])['id']
        until=time.time()+5
        while time.time()<until:
            judge_partial=Studio(base).judge_run(interrupted_judges)
            if len(judge_partial['items'])==1:break
            time.sleep(.01)
        assert len(judge_partial['items'])==1 and judge_partial['status']=='running'
        milestone_crash=create(mode='goal',max_steps=4);act(milestone_crash,'send','DURABLE_MILESTONE_GOAL')
        goal_partial=wait(milestone_crash,lambda state:state['status']=='running' and state['plan_revision']==1 and state['messages'][-1]['content']=='unfinished milestone')
        goal_checkpoint=Studio(base).session_plan(milestone_crash)
        assert goal_checkpoint['steps'][0]['status']=='completed' and goal_checkpoint['steps'][0]['id']
        bg_marker=pathlib.Path(tmp)/'background-crash-marker'
        bg_pidfile=pathlib.Path(tmp)/'background-crash-pid'
        import shlex
        orphan_bg=api(f'sessions/{bg}/background',dict(action='start',command='printf launched >> '+shlex.quote(str(bg_marker))+'; printf "%s" "$$" > '+shlex.quote(str(bg_pidfile))+'; sleep 30'))['id']
        until=time.time()+5
        while time.time()<until and not bg_pidfile.exists():time.sleep(.01)
        assert bg_marker.read_text()=='launched' and bg_pidfile.exists()
        orphan_pid=int(bg_pidfile.read_text())
        request_count=len(requests);process.kill();process.wait(timeout=5);process=start()
        matrix_interrupted=Studio(base).matrix_job(crash_matrix_job['id'])
        assert matrix_interrupted['status']=='interrupted' and matrix_interrupted['variants'][1]['run_id'] is None,matrix_interrupted
        assert Studio(base).run(matrix_owned_id)['status']=='interrupted'
        try:Studio(base).resume_matrix_job(crash_matrix_job['id']);raise AssertionError('interrupted provider call replayed')
        except EvaluationError as error:assert error.reason=='http_400'
        assert len(requests)==request_count
        print('PASS matrix SIGKILL recovery retains ownership, interrupts sequence and rejects inference replay',flush=True)
        orphan_receipt=api(f'sessions/{bg}/background',dict(action='output',task_id=orphan_bg))
        assert orphan_receipt['status']=='interrupted' and orphan_receipt['exit_code'] is None and orphan_receipt['duration_ms'] is None and orphan_receipt['finished_ms'] is None
        assert bg_marker.read_text()=='launched'
        try:os.killpg(orphan_pid,9)
        except ProcessLookupError:pass
        preserved_bg=api(f'sessions/{bg}/background',dict(action='output',task_id=bg_id))
        assert preserved_bg['status']=='completed' and preserved_bg['stdout']=='background-done' and preserved_bg['duration_ms'] is not None
        assert preserved_bg['name']=='Fixture background'
        assert api('sessions/'+bg)['background']['persistent'] is True
        assert Studio(base).cleanup_background(bg)['removed']>=3
        assert api('sessions/'+bg)['background']['tasks']==[]
        print('PASS background SIGKILL recovery retains completed output, interrupts unfinished ownership without replay and cleans receipts',flush=True)
        judge_recovered=Studio(base).judge_run(interrupted_judges)
        assert judge_recovered['status']=='interrupted' and judge_recovered['items']==judge_partial['items'] and judge_recovered['mean_score'] is None
        recovered_batch_trace=trace_rows('judge-run-'+interrupted_judges)[0]
        assert recovered_batch_trace['status']=='interrupted' and recovered_batch_trace['recovered_ms']


        recovered=api('evaluation/experiments/'+crash_run)
        assert recovered['status']=='interrupted' and recovered['recovered_ms'] and not recovered['strict_quality'] and recovered['mean_scores']=={}
        assert recovered['items'][0]==partial_run['items'][0]
        assert recovered['items'][1]['status']=='interrupted' and recovered['items'][1]['error']=='process_restart'
        for session_id,previous in [(crash_chat,chat_trace),('experiment-'+crash_run,exp_trace)]:
            repaired=trace_rows(session_id)[0];assert repaired['id']==previous['id'] and repaired['status']=='interrupted' and repaired['recovered_ms']
            assert not any(span['status']=='running' for span in repaired['spans'])
            for span in previous['spans']:
                after=next(row for row in repaired['spans'] if row['id']==span['id'])
                if span['status']=='running':assert after['status']=='interrupted' and after['duration_ms']==span['duration_ms']
                else:assert after==span
        assert stable_trace_path.read_bytes()==stable_trace and len(requests)==request_count
        process.terminate();process.wait(timeout=5);process=start()
        assert api('evaluation/experiments/'+crash_run)==recovered and len(requests)==request_count
        print('PASS SIGKILL recovery preserves finished outputs/usage, marks orphans interrupted with unknown durations, retains completed traces and never replays provider calls')
        retry_source=Studio(base).matrix_job(retry_source_job['id']);assert retry_source['status']=='interrupted'
        first_completed_id=retry_source['variants'][0]['run_id'];old_interrupted_id=retry_source['variants'][1]['run_id']
        assert Studio(base).run(first_completed_id)['status']=='completed'
        assert Studio(base).run(old_interrupted_id)['status']=='interrupted'
        retry_source_path=data/'evaluation'/'matrix-jobs'/(retry_source['id']+'.json');source_bytes=retry_source_path.read_bytes()
        before_explicit_retry=len(requests)
        retried_job=Studio(base).retry_matrix_job(retry_source['id']);assert retried_job['id']!=retry_source['id'] and retried_job['retry_of']==retry_source['id']
        retried_complete=wait_matrix_job(retried_job['id'])
        assert retried_complete['status']=='completed' and len(requests)==before_explicit_retry+2,retried_complete
        assert retried_complete['previous_run_ids']==[row['run_id'] for row in retry_source['variants']]
        assert retried_complete['variants'][0]['run_id']==first_completed_id
        assert retried_complete['variants'][1]['run_id']!=old_interrupted_id
        assert retry_source_path.read_bytes()==source_bytes and Studio(base).matrix_job(retry_source['id'])==retry_source
        assert Studio(base).run(old_interrupted_id)['status']=='interrupted'
        assert not Studio(base).experiment_matrix(retried_complete['matrix_id'])['passed']
        before_completed_retry=len(requests)
        try:Studio(base).retry_matrix_job(retried_complete['id']);raise AssertionError('completed job retried')
        except EvaluationError as error:assert error.reason=='http_400'
        assert len(requests)==before_completed_retry
        print('PASS explicit matrix retry creates distinct lineage, reuses completed baseline, calls only interrupted/unsent variants and preserves original history',flush=True)
        assert Studio(base).session_plan(milestone_manual)==manual_plan
        assert Studio(base).session_plan(milestone_crash)==goal_checkpoint
        assert api('sessions/'+milestone_crash)['status']=='paused'
        before_goal_resume=len(requests);act(milestone_crash,'resume')
        resumed_goal=wait(milestone_crash,lambda state:state['status']=='idle' and state['plan_revision']==2)
        assert len(requests)==before_goal_resume+2
        assert resumed_goal['plan'][0]==goal_checkpoint['steps'][0]
        assert all(step['status']=='completed' and step['acceptance'] and step['evidence'] for step in resumed_goal['plan'])
        resumed_checkpoint=Studio(base).session_plan(milestone_crash)
        assert resumed_checkpoint['checkpoints'][0]==goal_checkpoint['checkpoints'][0] and resumed_checkpoint['checkpoints'][1]['source']=='agent'
        print('PASS Goal checkpoint survives SIGKILL, does not auto-replay, and explicit resume receives stable completed milestone IDs and current revision',flush=True)
        if os.environ.get('ALLPAKA_TEST_NATIVE_VAULT')=='1':
            assert api('config')['credential_storage']
            dummy='allpaka-test-not-a-real-api-key'
            api('providers/openrouter/key',dict(key=dummy,persist=False))
            api('providers/openrouter/key',dict(key='',persist=True,use_current=True))
            public=api('config');assert dummy not in json.dumps(public)
            assert not any(dummy in p.read_text() for p in data.iterdir() if p.is_file())
            process.terminate();process.wait(timeout=5);process=start()
            router=next(p for p in api('config')['providers'] if p['id']=='openrouter')
            assert router['configured'] and router['saved'] and router['key_source']=='system'
            api('providers/openrouter/key',dict(key='',persist=False))
            process.terminate();process.wait(timeout=5);process=start()
            router=next(p for p in api('config')['providers'] if p['id']=='openrouter')
            assert not router['configured'] and not router['saved']
            print('PASS native vault save-current, restart loading, removal and no plaintext secrets')
    finally:
        if os.environ.get('ALLPAKA_TEST_NATIVE_VAULT')=='1':
            try:api('providers/openrouter/key',dict(key='',persist=False))
            except Exception:pass
        process.terminate();process.wait(timeout=5);log.close();mock.shutdown()
print('All Studio contract checks passed (mock provider, no live cloud requests).')
