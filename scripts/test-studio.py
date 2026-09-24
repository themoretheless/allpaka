#!/usr/bin/env python3
"""Studio end-to-end contract tests against a local streaming provider; no paid API calls."""
import json, os, pathlib, socket, subprocess, tempfile, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

requests = []
class Mock(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"mock","name":"Mock model","context_length":32000,"architecture":{"input_modalities":["text","image"]},"pricing":{"prompt":"0.000001","completion":"0.000002"},"supported_parameters":["tools"]}]}')
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(body)
        messages=body['messages'];last=messages[-1];content=last.get('content','')
        if isinstance(content,list):content=''.join(p.get('text','') for p in content)
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        def event(value):self.wfile.write(('data: '+json.dumps(value)+'\n\n').encode());self.wfile.flush()
        try:
            if messages[0].get('role')=='system' and messages[0].get('content','').startswith('Summarize conversation history'):
                if 'RETRY_COMPACT' in content: time.sleep(1.2)
                if 'FAIL_COMPACT' in content or ('RETRY_COMPACT' in content and body['max_tokens'] < 32768):
                    event({'choices':[{'delta':{'content':'incomplete'},'finish_reason':'length'}]})
                else:
                    event({'choices':[{'delta':{'content':'COMPACTED: Preserve user goals and completed work.'},'finish_reason':'stop'}]})
            elif 'STREAM_ANALYSIS' in content:
                event({'choices':[{'delta':{'reasoning_content':'Visible analysis <script>literal</script>'}}]})
                time.sleep(1)
                event({'choices':[{'delta':{'content':'Finished answer'},'finish_reason':'stop'}]})
            elif 'TOKEN_LIMIT_TOOL' in content:
                event({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'truncated','function':{'name':'write_file','arguments':'{"path":'}}]},'finish_reason':'length'}],'usage':{'completion_tokens':4096}})
            elif 'TOKEN_LIMIT_TEXT' in content:
                time.sleep(.15)
                event({'choices':[{'delta':{'content':'partial text'},'finish_reason':'length'}],'usage':{'completion_tokens':4096}})
            elif 'SLOW' in content:
                event({'choices':[{'delta':{'content':'started '}}]})
                for _ in range(15):time.sleep(.1);event({'choices':[{'delta':{'content':'.'}}]})
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
            elif 'Ты — участник swarm' in content and 'SWARM-FAIL' in content and '«risks»' in content:
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
    env=dict(os.environ,OPENROUTER_API_KEY='',ALLPAKA_LOCAL_BASE_URL=f'http://127.0.0.1:{mock.server_port}/v1')
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
    def wait(id,predicate,timeout=8):
        until=time.time()+timeout
        while time.time()<until:
            s=api('sessions/'+id)
            if predicate(s):return s
            time.sleep(.03)
        raise AssertionError(s)
    settings=dict(project_id='default',provider='local',model='mock',mode='chat',max_steps=8,allow_writes=False)
    def create(**kw):return api('sessions',dict(settings,**kw))['id']
    def act(id,kind,text='',**kw):return api(f'sessions/{id}/actions',dict(kind=kind,text=text,**kw))
    process=start()
    try:
        assert len(api('config')['providers'])==8
        catalog=api('providers/local/models')['catalog'][0]
        assert catalog['context_length']==32000 and catalog['pricing']['prompt']=='0.000001'
        assert catalog['architecture']['input_modalities']==['text','image']
        assert api('providers/local/models')['models']==['mock']
        # Origin and intent protections.
        for headers in [{'Content-Type':'application/json'},{'Content-Type':'application/json','X-Allpaka-Client':'studio','Origin':'https://elsewhere.example'}]:
            try:api('sessions',settings,headers);raise AssertionError('unsafe request accepted')
            except urllib.error.HTTPError as e:assert e.code==403
        id=create();act(id,'send','hello');s=wait(id,lambda s:s['status']=='idle' and len(s['messages'])==2);assert 'hello' in s['messages'][-1]['content']
        print('PASS chat, model catalog, usage, request guards')
        branch=api(f'sessions/{id}/branch',dict(message_count=2))['id']
        branched=api('sessions/'+branch)
        assert branched['messages']==s['messages'] and branched['parent']==dict(session_id=id,message_count=2)
        assert branched['queue']==[] and branched['status']=='idle'
        act(branch,'send','BRANCH-ONLY');wait(branch,lambda x:x['status']=='idle' and len(x['messages'])==4)
        assert len(api('sessions/'+id)['messages'])==2
        empty=api(f'sessions/{id}/branch',dict(message_count=0))['id'];assert api('sessions/'+empty)['messages']==[]
        for count in [1,3]:
            try:api(f'sessions/{id}/branch',dict(message_count=count));raise AssertionError('invalid boundary accepted')
            except urllib.error.HTTPError as e:assert e.code==400
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
        read_only_auto=create(mode='auto',allow_writes=True);act(read_only_auto,'send','CHECK-CAPABILITIES')
        wait(read_only_auto,lambda s:s['status']=='idle' and len(s['messages'])==2)
        tool_names=[t['function']['name'] for t in requests[-1]['tools']]
        assert 'read_file' in tool_names and 'write_file' not in tool_names and 'edit_file' not in tool_names
        assert 'Tools actually available for THIS request:' in requests[-1]['messages'][0]['content']
        project=api('projects',dict(id='default',name='Test project',instructions='',roots=[dict(alias='workspace',path=str(workspace),writable=True),dict(alias='reference',path=str(reference),writable=False)]))
        planned=create(mode='plan',allow_writes=True);act(planned,'send','TRY_WRITE');s=wait(planned,lambda s:s['status']=='idle' and len(s['messages'])>=4);assert not (workspace/'result.txt').exists();assert 'requires Auto' in next(m['content'] for m in s['messages'] if m['role']=='tool')
        auto=create(mode='auto',allow_writes=True);act(auto,'send','TRY_WRITE');wait(auto,lambda s:s['status']=='idle' and len(s['messages'])>=4);assert (workspace/'result.txt').read_text()=='verified write'
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
        compacted=create(auto_compact=False)
        for i in range(4):
            act(compacted,'send',f'long turn {i} '+('context '*1000))
            wait(compacted,lambda s:s['status']=='idle' and len(s['messages'])==(i+1)*2)
        original=api('sessions/'+compacted)['messages']
        act(compacted,'compact')
        summary=wait(compacted,lambda s:s['status']=='idle' and s.get('compaction'))
        assert summary['messages']==original and summary['compaction']['through']==4
        stats=summary['context_stats']
        assert stats['compacted_messages']==4 and stats['messages']==8
        assert stats['estimated_history_tokens'] < stats['original_history_tokens']
        assert stats['remaining_before_compact']==max(0,stats['compact_threshold']-stats['estimated_history_tokens'])
        act(compacted,'send','AFTER-COMPACT')
        wait(compacted,lambda s:s['status']=='idle' and len(s['messages'])==10)
        assert 'COMPACTED:' in requests[-1]['messages'][1]['content']
        assert not any('long turn 0' in m.get('content','') for m in requests[-1]['messages'])
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
        print('PASS compaction retries identical input with larger budget and preserves history')
        automatic=create(auto_compact=True,compact_threshold=4096)
        for i in range(4):
            act(automatic,'send',f'auto turn {i} '+('context '*1000))
            wait(automatic,lambda s:s['status']=='idle' and len(s['messages'])==(i+1)*2)
        assert api('sessions/'+automatic)['compaction']['through']>=2
        failed=create(auto_compact=False)
        for i in range(3):
            act(failed,'send','FAIL_COMPACT '+str(i))
            wait(failed,lambda s:s['status']=='idle' and len(s['messages'])==(i+1)*2)
        before=api('sessions/'+failed)['messages'];act(failed,'compact')
        unchanged=wait(failed,lambda s:s['status']=='error')
        assert unchanged['compaction'] is None and unchanged['messages']==before
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
        imported=api('sessions/import',dict(session=exported,project_id='default'))['id']
        imported_state=api('sessions/'+imported)
        assert imported!=edited and imported_state['messages']==exported['messages']
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
        assert sorted(member_tools)==['list_files','read_file'] and 'write_file' not in member_tools
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
        print('PASS swarm member reports its tool-loop step as a field beside the status word')
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
        print('PASS swarm names a failed member instead of inventing its report')
        critic=create(mode='swarm',swarm=dict(members=two_members,max_steps_per_member=1,critic=True))
        act(critic,'send','SWARM-CRITIC')
        state=wait(critic,lambda s:s['status']=='idle' and s['messages'][-1].get('swarm'))
        assert state['messages'][-1]['content']=='MASTER merged answer (critic passed)'
        assert len([r for r in requests if 'Ты — критик swarm-результата' in str(r['messages'][-1]['content'])])==1
        assert state['usage']['completion_tokens']==6+6+8+10
        print('PASS swarm critic pass replaces the draft only after it succeeds')
        stopping=create(mode='swarm',swarm=dict(members=[dict(label='slow',role='collect',provider='local',model='mock'),dict(label='risks',role='regressions',provider='local',model='mock')],max_steps_per_member=1))
        act(stopping,'send','SWARM-STOP')
        wait(stopping,lambda s:s['status']=='running' and 'REPORT from slow' in json.dumps(s['messages'][-1].get('swarm') or []))
        act(stopping,'stop')
        stopped=wait(stopping,lambda s:s['status']=='paused')
        statuses=[r['status'] for r in stopped['messages'][-1]['swarm']]
        assert not any(status=='running' or status=='queued' for status in statuses), statuses
        assert 'cancelled' in statuses and stopped['messages'][-1]['content']==''
        print('PASS stop cancels the whole wave and marks unfinished reports honestly')
        swarm_export=api('sessions/'+swarm)
        swarm_import=api('sessions/import',dict(session=swarm_export,project_id='default'))['id']
        imported_swarm=api('sessions/'+swarm_import)
        assert imported_swarm['messages']==swarm_export['messages'] and imported_swarm['settings']['mode']=='chat'
        print('PASS swarm reports survive export/import without re-running the wave')
        # Persisted projects and conversations survive restart.
        process.terminate();process.wait(timeout=5);process=start();assert api('sessions/'+id)['messages'];assert len(api('config')['projects'][0]['roots'])==2
        assert api('sessions/'+vision)['messages'][0]['images'][0]['name']=='tiny.png'
        assert api('sessions/'+branch)['parent']['session_id']==id
        assert api('sessions/'+compacted)['compaction']==summary['compaction']
        assert api('sessions/'+imported)['folder']=='trash'
        assert api('sessions/'+swarm)['messages'][-1]['swarm'][0]['content']=='REPORT from scout'
        assert api('sessions/'+compacted)['title']=='Renamed conversation'
        print('PASS conversation, image, compaction and project persistence')
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
