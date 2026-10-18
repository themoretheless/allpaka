#!/usr/bin/env python3
"""Studio harness end to end against a local streaming provider; no paid API calls.

Covers the editable harness (validation, activation, load_skill, hooks), a
JSON Propose-Probe-Compose job (probes on copies, adoption), an agentic job
over a stream of two tasks with a report-only rubric, and the experience
library, through the Python SDK.
"""
import json, os, pathlib, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sdk/python"))
from allpaka_studio import Studio, EvaluationError

SKILL = "---\nname: note\ndescription: the task asks for a note file\n---\nWrite workspace/out.txt with the note, then reread it.\n"
HOOKS = json.dumps({"post_tool": [{"matcher": "write_file", "inject": "Reopen what you wrote."}]})
MEMORY = 'In this workspace notes go to workspace/out.txt.'
systems = []
external_requests = []
CODEX_BIN = os.environ.get('ALLPAKA_CODEX_BIN')
OPENCODE_BIN = os.environ.get('ALLPAKA_OPENCODE_BIN')

class Mock(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[{"id":"mock"}]}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path.endswith('/responses') or body.get('model') == 'ocmock':
            return self.external(body)
        messages = body['messages']; system = messages[0].get('content', ''); systems.append(system)
        tools = [t['function']['name'] for t in body.get('tools') or []]
        results = [m['content'] for m in messages if m.get('role') == 'tool']
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        def say(text): return {'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}]}
        def call(name, args): return {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': f'c{len(results)}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'finish_reason': 'tool_calls'}]}
        proposer = 'YOUR DIMENSION' in system or 'composing THE final harness' in system
        if proposer and tools:
            # Agentic proposers: edit the `harness` root with file tools.
            assert 'run_command' not in tools and 'load_skill' not in tools
            if 'YOUR DIMENSION: INV' in system:
                event = call('write_file', {'path': 'harness/memory.md', 'content': MEMORY}) if not results else say('base1 searched for the target file')
            elif 'YOUR DIMENSION: DOM' in system:
                event = say('KEEP_BASE: nothing domain specific')
            elif not results:
                event = call('read_file', {'path': 'context/pool/inv1/harness/memory.md'})
            elif len(results) == 1:
                assert MEMORY in results[0], results[0]
                event = call('write_file', {'path': 'harness/memory.md', 'content': MEMORY})
            else:
                event = say('kept -> memory.md -> base1 searched 1/1')
        elif proposer:
            if 'YOUR DIMENSION: INV' in system:
                event = say(json.dumps({'files': {'memory.md': MEMORY}, 'notes': 'base1 searched for the target file'}))
            elif 'YOUR DIMENSION: DOM' in system:
                event = say(json.dumps({'files': {}, 'notes': 'KEEP_BASE: nothing domain specific'}))
            else:
                assert 'candidate inv1' in messages[-1]['content'] and 'candidate dom1' in messages[-1]['content']
                event = say(json.dumps({'adopt': 'inv1', 'notes': 'kept -> memory.md -> base1 searched 1/1'}))
        elif not tools:
            if 'complete record' in system:
                event = say(json.dumps({'edits': [{'component': 'memory.md', 'candidates': 'inv1', 'change': 'names the note path', 'observed': 'agg1 wrote directly', 'verdict': 'confounded'}],
                                        'experience': {'cue': 'a task writes a note file', 'lesson': 'state the target path in memory.md', 'evidence': 'inv1 vs base1', 'confidence': 'low: one rollout'}}))
            elif system.startswith('Evaluate the supplied answer'):
                reference = json.loads(messages[-1]['content'])['reference']
                event = say(json.dumps({'score': 1.0 if reference == 'a note file exists' else 0.75, 'reason': 'note written'}))
            else:
                event = say('unexpected')
        elif not results:
            assert 'load_skill' in tools and '# Available skills' in system and 'load the `note` skill' in system
            event = call('load_skill', {'name': 'note'})
        elif len(results) == 1:
            assert 'Write workspace/out.txt' in results[0]
            event = call('write_file', {'path': 'workspace/out.txt', 'content': 'note'})
        else:
            assert '[harness hook] Reopen what you wrote.' in results[1]
            event = say('wrote the note')
        self.wfile.write(('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()); self.wfile.flush()

    def external(self, body):
        """Real Codex (Responses API) and OpenCode (chat completions) runs."""
        external_requests.append(json.dumps(body))
        text = 'external-real-ok'
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        if self.path.endswith('/responses'):
            item = {'id': 'm1', 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}
            events = [('response.created', {'type': 'response.created', 'response': {'id': 'r1', 'object': 'response', 'status': 'in_progress', 'output': []}}),
                      ('response.output_item.added', {'type': 'response.output_item.added', 'output_index': 0, 'item': dict(item, status='in_progress', content=[])}),
                      ('response.output_text.delta', {'type': 'response.output_text.delta', 'item_id': 'm1', 'output_index': 0, 'content_index': 0, 'delta': text}),
                      ('response.output_item.done', {'type': 'response.output_item.done', 'output_index': 0, 'item': item}),
                      ('response.completed', {'type': 'response.completed', 'response': {'id': 'r1', 'object': 'response', 'status': 'completed', 'output': [item], 'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2, 'input_tokens_details': {'cached_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}}}})]
            for name, data in events:
                self.wfile.write(f'event: {name}\ndata: {json.dumps(data)}\n\n'.encode())
        else:
            for chunk in [{'id': 'c1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'ocmock', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}, 'finish_reason': None}]},
                          {'id': 'c1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'ocmock', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}]:
                self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
            self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()

mock = ThreadingHTTPServer(('127.0.0.1', 0), Mock); threading.Thread(target=mock.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory(prefix='allpaka-harness-test-') as tmp:
    root = pathlib.Path(tmp); workspace = root / 'workspace'; workspace.mkdir(); (workspace / 'readme.txt').write_text('fixture')
    data = root / 'history'; data.mkdir()
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close(); base = f'http://127.0.0.1:{port}'
    env = dict(os.environ, ALLPAKA_ONLINE_WORKER='0', OPENROUTER_API_KEY='', ALLPAKA_LOCAL_BASE_URL=f'http://127.0.0.1:{mock.server_port}/v1')
    binary = pathlib.Path(os.environ.get('ALLPAKA_TEST_BINARY', str(pathlib.Path(__file__).resolve().parents[1] / 'target/debug/allpaka')))
    log = open(root / 'server.log', 'w')
    def api(path, body=None):
        req = urllib.request.Request(base + '/api/' + path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'X-Allpaka-Client': 'studio'})
        with urllib.request.urlopen(req, timeout=5) as r: return json.load(r)
    def reject(path, body, needle):
        try: api(path, body); raise AssertionError('request unexpectedly accepted')
        except urllib.error.HTTPError as error: assert error.code == 400 and needle in json.load(error)['error']
    def until(fetch, predicate, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = fetch()
            if predicate(value): return value
            time.sleep(.05)
        raise AssertionError(value)
    process = subprocess.Popen([str(binary), 'studio', '--bind', f'127.0.0.1:{port}', '--workspace', str(workspace), '--data-dir', str(data)], env=env, stdout=log, stderr=log)
    try:
        for _ in range(200):
            try: api('config'); break
            except Exception:
                if process.poll() is not None: raise RuntimeError((root / 'server.log').read_text())
                time.sleep(.03)
        studio = Studio(base)
        project = api('projects', dict(id='', name='harness', roots=[dict(alias='workspace', path=str(workspace), writable=True)], instructions=''))['id']

        empty = studio.harness(project)
        assert empty['active'] is None and empty['current']['tree'] == {} and empty['current']['preview'] == ''
        reject('harness', dict(project_id=project, tree={'notes.txt': 'x'}), 'not a harness path')
        reject('harness', dict(project_id=project, tree={'skills/note/SKILL.md': '---\nname: other\n---\n'}), 'frontmatter name must be `note`')
        reject('harness', dict(project_id=project, tree={'hooks.json': '{"pre_tool":[{"inject":"x"}]}'}), 'Each pre_tool hook needs exactly one of block, command')
        saved = studio.save_harness(project, {'skills/note/SKILL.md': SKILL, 'hooks.json': HOOKS}, note='fixture', activate=True)
        assert 'load the `note` skill' in saved['current']['tree']['guardrails.md'], 'trigger line is added on save'
        first = studio.harness(project)
        assert first['active'] == saved['revision']['id'] and len(first['revisions']) == 1

        settings = dict(project_id=project, provider='local', model='mock', mode='auto', max_steps=8, allow_writes=True)
        session = api('sessions', settings)['id']
        api(f'sessions/{session}/actions', dict(kind='send', text='Leave a note', settings=settings))
        done = until(lambda: api('sessions/' + session), lambda s: s['status'] in ('idle', 'error'))
        assert done['status'] == 'idle', done.get('error')
        assert done['messages'][-1]['content'] == 'wrote the note'
        assert (workspace / 'out.txt').read_text() == 'note'
        (workspace / 'out.txt').unlink()
        stored = sorted(p.name for p in data.glob('*.json'))

        # One JSON-mode step on the conversation's task.
        job = studio.start_harness_job(session, k=1, mode='parallel', agentic=False)
        detail = until(lambda: studio.harness_job(job['id']), lambda d: d['job']['status'] in ('completed', 'failed'), 60)
        assert detail['job']['status'] == 'completed', detail['job'].get('error')
        result = detail['job']['results'][0]; outcome = result['outcome']
        assert (outcome['rollouts'], outcome['stages'], outcome['chosen']) == (4, 3, 'agg1')
        assert all(run['delivered'] for run in outcome['runs']), outcome['runs']
        assert detail['diffs'][0]['added'] == ['memory.md'] and detail['active'] == first['active']
        assert not (workspace / 'out.txt').exists(), 'probes run on copies'
        assert sorted(p.name for p in data.glob('*.json')) == stored, 'probe and proposer sessions are not saved'
        assert any('# Long-term memory\n' + MEMORY in s for s in systems), 'probes run the candidate harness'
        studio.adopt_harness_job(job['id'])
        adopted = studio.harness(project)
        assert adopted['active'] == result['revision'] and adopted['current']['tree']['memory.md'] == MEMORY

        # Agentic proposers over a stream of two tasks, with a report-only rubric.
        job = studio.start_harness_job(session, k=1, agentic=True, rubric='The note file exists.', tasks=['Leave a note', 'Leave another note'])
        detail = until(lambda: studio.harness_job(job['id']), lambda d: d['job']['status'] in ('completed', 'failed'), 90)
        assert detail['job']['status'] == 'completed', detail['job'].get('error')
        results = detail['job']['results']
        assert [r['task'] for r in results] == ['Leave a note', 'Leave another note']
        for r in results:
            o = r['outcome']
            assert (o['rollouts'], o['stages'], o['chosen'], o['fallbacks']) == (4, 3, 'agg1', []), o
            assert o['scores'] == {'base1': 0.75, 'agg1': 0.75}, o['scores']
            assert r['revision'] is None, 'the composer kept memory.md, which the adopted base already has'
        assert any('YOURS TO EDIT' in s for s in systems) and any('HOW TO COMPOSE' in s for s in systems)
        assert not (workspace / 'out.txt').exists()

        # A stream from a dataset: the expected output reaches only the judge.
        dataset = api('evaluation/datasets', dict(project_id=project, name='Notes', base_version=0, samples=[dict(id='n1', input='Leave a note', expected_output='a note file exists')]))
        job = studio.start_harness_job(session, agentic=False, rubric='The note file exists.', dataset=dict(id=dataset['id']))
        detail = until(lambda: studio.harness_job(job['id']), lambda d: d['job']['status'] in ('completed', 'failed'), 60)
        assert detail['job']['status'] == 'completed', detail['job'].get('error')
        assert detail['job']['tasks'] == ['Leave a note'] and detail['job']['references'] == ['a note file exists']
        assert detail['job']['results'][0]['outcome']['scores'] == {'base1': 1.0, 'agg1': 1.0}
        assert not any('a note file exists' in s for s in systems if 'YOUR DIMENSION' in s or 'composing' in s)

        # An external agent CLI executes the rollouts on the materialized harness.
        job = studio.start_harness_job(session, agentic=False, external=dict(command="cat > task.txt; test -f .agents/skills/note/SKILL.md && echo external-ok", layout='codex'))
        detail = until(lambda: studio.harness_job(job['id']), lambda d: d['job']['status'] in ('completed', 'failed'), 60)
        assert detail['job']['status'] == 'completed', detail['job'].get('error')
        runs = detail['job']['results'][0]['outcome']['runs']
        assert all(r['delivered'] for r in runs), runs
        assert not (workspace / 'task.txt').exists()

        # Optional: the real Codex and OpenCode CLIs, pointed at this mock.
        expected_library = 5
        clis = []
        if CODEX_BIN:
            home = root / 'codex-home'; home.mkdir()
            provider = f'model_providers.mock={{name="mock",base_url="http://127.0.0.1:{mock.server_port}/v1",wire_api="responses",env_key="MOCK_KEY"}}'
            clis.append(('codex', f"CODEX_HOME={home} MOCK_KEY=x {CODEX_BIN} exec --skip-git-repo-check -c model_provider=mock -c '{provider}' -m cxmock -"))
        if OPENCODE_BIN:
            home = root / 'oc-home'; home.mkdir()
            config = json.dumps({'provider': {'mock': {'npm': '@ai-sdk/openai-compatible', 'name': 'mock', 'options': {'baseURL': f'http://127.0.0.1:{mock.server_port}/v1', 'apiKey': 'x'}, 'models': {'ocmock': {'name': 'ocmock'}}}}, 'model': 'mock/ocmock'})
            clis.append(('opencode', f"HOME={home} XDG_CONFIG_HOME={home}/c XDG_DATA_HOME={home}/d XDG_CACHE_HOME={home}/x OPENCODE_CONFIG_CONTENT='{config}' {OPENCODE_BIN} run"))
        for layout, command in clis:
            external_requests.clear()
            job = studio.start_harness_job(session, agentic=False, external=dict(command=command, layout=layout))
            detail = until(lambda: studio.harness_job(job['id']), lambda d: d['job']['status'] in ('completed', 'failed'), 300)
            assert detail['job']['status'] == 'completed', detail['job'].get('error')
            runs = detail['job']['results'][0]['outcome']['runs']
            assert all(r['delivered'] for r in runs), (layout, runs)
            seen = [r for r in external_requests if MEMORY in r]
            assert seen, f'{layout} never sent the harness to the model'
            assert any('the task asks for a note' in r for r in seen), f'{layout} did not discover the skill'
            expected_library += 1
            print(f'real {layout}: {len(external_requests)} model requests, harness and skill seen')

        library = studio.harness_library(project)['entries']
        assert len(library) == expected_library and all(e['entry']['cue'] == 'a task writes a note file' for e in library)
        assert '[confounded]' in library[0]['experience']
        try: studio.start_harness_job(session, k=9); raise AssertionError('k=9 accepted')
        except EvaluationError as error: assert str(error) == 'http_400'
        reject('harness/jobs', dict(session_id=session, tasks=['ok', '   ']), 'at most 20 non-empty tasks')
        studio.activate_harness(project, None)
        assert studio.harness(project)['active'] is None
        print('studio harness: ok')
    finally:
        process.terminate(); process.wait(timeout=10)
