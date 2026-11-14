#!/usr/bin/env python3
"""Named sessions (`session_id`) beside one another in `allpaka serve`.

    scripts/check-named-sessions.py --addr 127.0.0.1:8099

1. Two long requests with different session ids: the second must stream
   while the first still runs (its first token well before the first ends).
2. The first session was kept: its first message sent again reuses the
   session's KV (`usage.prompt_tokens_details.cached_tokens`). (A longer
   follow-up than the session was sized for resets it; see inference.rs.)
3. A request for a session that is running (the second) waits for it: it completes after
   that session's request does.
Exits non-zero on any failure.
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request


def post(addr, body, on_token=None):
    req = urllib.request.Request(f'http://{addr}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not body.get('stream'):
            return json.loads(r.read())
        text = ''
        for line in r:
            if line.startswith(b'data: {'):
                event = json.loads(line[6:])
                if 'error' in event:
                    raise RuntimeError(event['error'])
                delta = event['choices'][0].get('delta', {}).get('content')
                if delta:
                    text += delta
                    if on_token:
                        on_token()
        return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--addr', default='127.0.0.1:8099')
    a = p.parse_args()
    filler = ' '.join(f'w{i}' for i in range(1500))
    times = {}
    history = {}

    def run(tag, sid, started):
        first = []

        def tick():
            if not first:
                first.append(time.perf_counter())
                started.set()
        sent = time.perf_counter()
        prompt = f'My name is {tag}. Write a long story:\n' + filler
        text = post(a.addr, {'session_id': sid, 'stream': True, 'temperature': 0, 'max_tokens': 400,
                             'messages': [{'role': 'user', 'content': prompt}]}, tick)
        times[tag] = (sent, first[0] if first else None, time.perf_counter())
        history[tag] = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': text}]

    sa, sb = threading.Event(), threading.Event()
    ta = threading.Thread(target=run, args=('Alice', 'sess-a', sa))
    tb = threading.Thread(target=run, args=('Bob', 'sess-b', sb))
    ta.start()
    sa.wait(120)
    tb.start()
    # 3: while both run, a request for Bob's running session waits for it.
    time.sleep(0.3)
    busy = {}

    def same_session():
        try:
            post(a.addr, {'session_id': 'sess-c', 'max_tokens': 4,
                          'messages': [{'role': 'user', 'content': 'hi'}]})
            post(a.addr, {'session_id': 'sess-b', 'max_tokens': 4,
                          'messages': [{'role': 'user', 'content': 'hi'}]})
            busy['done'] = time.perf_counter()
        except urllib.error.HTTPError as e:
            busy['status'] = e.code
    tc = threading.Thread(target=same_session)
    tc.start()
    ta.join()
    tb.join()
    tc.join()
    a_sent, a_first, a_done = times['Alice']
    b_sent, b_first, b_done = times['Bob']
    overlap = b_first is not None and b_first < a_done
    print(f'Alice: first token {1000 * (a_first - a_sent):.0f} ms, done {1000 * (a_done - a_sent):.0f} ms')
    print(f'Bob:   first token {1000 * (b_first - b_sent):.0f} ms (before Alice finished: {overlap})')
    waited = 'done' in busy and busy['done'] > b_done
    print(f'request for the running session: {busy} (completed after it: {waited})')
    # 2: the first session was kept.
    reply = post(a.addr, {'session_id': 'sess-a', 'temperature': 0, 'max_tokens': 16,
                          'messages': history['Alice'][:1]})
    cached = reply['usage']['prompt_tokens_details']['cached_tokens']
    print(f'sess-a again: cached {cached} of {reply["usage"]["prompt_tokens"]} prompt tokens')
    ok = overlap and waited and cached > 1000
    print('OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
