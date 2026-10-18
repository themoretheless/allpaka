#!/usr/bin/env python3
"""Studio harness end to end against a local streaming provider; no paid API calls.

Covers the editable harness (validation, activation, load_skill, hooks), a
Propose-Probe-Compose job (probes on copies, adoption) and the experience
library.
"""
import json, os, pathlib, socket, subprocess, tempfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SKILL = "---\nname: note\ndescription: the task asks for a note file\n---\nWrite workspace/out.txt with the note, then reread it.\n"
HOOKS = json.dumps({"post_tool": [{"matcher": "write_file", "inject": "Reopen what you wrote."}]})
systems = []

class Mock(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[{"id":"mock"}]}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body['messages']; system = messages[0].get('content', ''); systems.append(system)
        tools = [t['function']['name'] for t in body.get('tools') or []]
        results = [m['content'] for m in messages if m.get('role') == 'tool']
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        def say(text): return {'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}]}
        def call(name, args): return {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': f'c{len(results)}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'finish_reason': 'tool_calls'}]}
        if not tools:
            if 'YOUR DIMENSION: INV' in system:
                event = say(json.dumps({'files': {'memory.md': 'In this workspace notes go to workspace/out.txt.'}, 'notes': 'base1 searched for the target file'}))
            elif 'YOUR DIMENSION: DOM' in system:
                event = say(json.dumps({'files': {}, 'notes': 'KEEP_BASE: nothing domain specific'}))
            elif 'composing THE final harness' in system:
                assert 'candidate inv1' in messages[-1]['content'] and 'candidate dom1' in messages[-1]['content']
                event = say(json.dumps({'adopt': 'inv1', 'notes': 'kept -> memory.md -> base1 searched 1/1'}))
            elif 'complete record' in system:
                event = say(json.dumps({'edits': [{'component': 'memory.md', 'candidates': 'inv1', 'change': 'names the note path', 'observed': 'agg1 wrote directly', 'verdict': 'confounded'}],
                                        'experience': {'cue': 'a task writes a note file', 'lesson': 'state the target path in memory.md', 'evidence': 'inv1 vs base1', 'confidence': 'low: one rollout'}}))
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
        project = api('projects', dict(id='', name='harness', roots=[dict(alias='workspace', path=str(workspace), writable=True)], instructions=''))['id']

        empty = api('harness?project_id=' + project)
        assert empty['active'] is None and empty['current']['tree'] == {} and empty['current']['preview'] == ''
        reject('harness', dict(project_id=project, tree={'notes.txt': 'x'}), 'not a harness path')
        reject('harness', dict(project_id=project, tree={'skills/note/SKILL.md': '---\nname: other\n---\n'}), 'frontmatter name must be `note`')
        saved = api('harness', dict(project_id=project, tree={'skills/note/SKILL.md': SKILL, 'hooks.json': HOOKS}, note='fixture', activate=True))
        assert 'load the `note` skill' in saved['current']['tree']['guardrails.md'], 'trigger line is added on save'
        first = api('harness?project_id=' + project)
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

        job = api('harness/jobs', dict(session_id=session, k=1, mode='parallel'))
        detail = until(lambda: api('harness/jobs/' + job['id']), lambda d: d['job']['status'] in ('completed', 'failed'), 60)
        assert detail['job']['status'] == 'completed', detail['job'].get('error')
        outcome = detail['job']['outcome']
        assert (outcome['rollouts'], outcome['stages'], outcome['chosen']) == (4, 3, 'agg1')
        assert all(run['delivered'] for run in outcome['runs']), outcome['runs']
        assert detail['diff']['added'] == ['memory.md'] and detail['active'] == first['active']
        assert not (workspace / 'out.txt').exists(), 'probes run on copies'
        assert sorted(p.name for p in data.glob('*.json')) == stored, 'probe sessions are not saved'
        assert any('# Long-term memory\nIn this workspace notes go to workspace/out.txt.' in s for s in systems), 'probes run the candidate harness'

        api(f"harness/jobs/{job['id']}/adopt", {})
        adopted = api('harness?project_id=' + project)
        assert adopted['active'] == detail['job']['revision'] and 'memory.md' in adopted['current']['tree']
        library = api('harness/library?project_id=' + project)['entries']
        assert len(library) == 1 and library[0]['entry']['cue'] == 'a task writes a note file' and '[confounded]' in library[0]['experience']
        reject('harness/jobs', dict(session_id=session, k=9), 'k must be between 1 and 4')
        api('harness/activate', dict(project_id=project, revision=None))
        assert api('harness?project_id=' + project)['active'] is None
        print('studio harness: ok')
    finally:
        process.terminate(); process.wait(timeout=10)
