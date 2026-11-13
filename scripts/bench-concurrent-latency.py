#!/usr/bin/env python3
"""Decode latency of one chat stream while another request competes for the engine.

    scripts/bench-concurrent-latency.py --addr 127.0.0.1:8080 [--model ID]
        [--prompt-tokens 2000] [--new-tokens 64] [--json out.json]

Two scenarios against a running `allpaka serve`, each with a short streaming
request (the "chat") and a heavy one (long prompt, same number of new tokens):

  chat-first   the chat is already streaming when the heavy request arrives
  heavy-first  the heavy request is already running when the chat arrives

For the chat it reports time to first token and the largest and median gap
between tokens; for the heavy request its total time. A scheduler that
interleaves work bounds the chat's numbers in both scenarios; one that runs
requests one after another does not bound `heavy-first`.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def post(addr, body, on_chunk=None):
    request = urllib.request.Request(
        f'http://{addr}/v1/chat/completions', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=3600) as response:
        if on_chunk is None:
            return json.loads(response.read())
        for line in response:
            if line.startswith(b'data: {'):
                on_chunk(json.loads(line[6:]))


def chat(addr, model, new_tokens, started):
    """Stream a short request; return the arrival time of every content delta."""
    arrivals = []
    sent = time.perf_counter()

    def on_chunk(event):
        if event['choices'][0].get('delta', {}).get('content'):
            arrivals.append(time.perf_counter())
            started.set()

    post(addr, {'model': model, 'stream': True, 'temperature': 0, 'max_tokens': new_tokens,
                'messages': [{'role': 'user', 'content': 'Count from one to one hundred, slowly.'}]},
         on_chunk)
    started.set()
    return sent, arrivals


def heavy(addr, model, prompt_tokens, new_tokens, started):
    # About one token per word for this filler in the tokenizers serve supports.
    filler = ' '.join(f'item{i}' for i in range(prompt_tokens // 2))
    sent = time.perf_counter()
    started.set()
    reply = post(addr, {'model': model, 'temperature': 0, 'max_tokens': new_tokens,
                        'messages': [{'role': 'user', 'content': f'Summarise this list: {filler}'}]})
    return sent, time.perf_counter(), reply.get('usage', {})


def scenario(name, addr, model, prompt_tokens, new_tokens):
    result = {}
    chat_started, heavy_started = threading.Event(), threading.Event()

    def run_chat():
        result['chat'] = chat(addr, model, new_tokens, chat_started)

    def run_heavy():
        result['heavy'] = heavy(addr, model, prompt_tokens, new_tokens, heavy_started)

    first, second, gate = ((run_chat, run_heavy, chat_started) if name == 'chat-first'
                           else (run_heavy, run_chat, heavy_started))
    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    # The second request leaves once the first is demonstrably under way: its
    # first token for the chat, or shortly after sending for the heavy one.
    gate.wait(timeout=600)
    if name == 'heavy-first':
        time.sleep(0.2)
    b.start()
    a.join()
    b.join()

    sent, arrivals = result['chat']
    gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
    heavy_sent, heavy_done, usage = result['heavy']
    return {
        'scenario': name,
        'chat_deltas': len(arrivals),
        'chat_first_token_ms': round((arrivals[0] - sent) * 1000, 1) if arrivals else None,
        'chat_max_gap_ms': round(max(gaps) * 1000, 1) if gaps else None,
        'chat_median_gap_ms': round(statistics.median(gaps) * 1000, 1) if gaps else None,
        'heavy_total_ms': round((heavy_done - heavy_sent) * 1000, 1),
        'heavy_prompt_tokens': usage.get('prompt_tokens'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--addr', default='127.0.0.1:8080')
    p.add_argument('--model', default='')
    p.add_argument('--prompt-tokens', type=int, default=2000)
    p.add_argument('--new-tokens', type=int, default=64)
    p.add_argument('--json')
    a = p.parse_args()
    rows = [scenario(name, a.addr, a.model, a.prompt_tokens, a.new_tokens)
            for name in ('chat-first', 'heavy-first')]
    for row in rows:
        print(f"{row['scenario']:<12} chat: first token {row['chat_first_token_ms']} ms, "
              f"max gap {row['chat_max_gap_ms']} ms, median gap {row['chat_median_gap_ms']} ms "
              f"({row['chat_deltas']} deltas); heavy: {row['heavy_total_ms']} ms "
              f"for {row['heavy_prompt_tokens']} prompt tokens")
    if a.json:
        with open(a.json, 'w') as out:
            json.dump({'addr': a.addr, 'prompt_tokens': a.prompt_tokens,
                       'new_tokens': a.new_tokens, 'scenarios': rows}, out, indent=2)
            out.write('\n')


if __name__ == '__main__':
    main()
