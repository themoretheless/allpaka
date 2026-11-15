#!/usr/bin/env python3
"""Decode rate as a function of how much context is already in the KV cache.

    scripts/bench-decode-vs-context.py --allpaka <allpaka binary> --llama-server <llama-server binary>
        --model <model.gguf> [--contexts 1000 4000 8000 12000] [--new-tokens 64] [--json out.json]

For each context size the same prompt goes to `allpaka serve` and to
`llama-server`, greedy, and each engine's own decode rate is read back:
allpaka's from the `timing` it reports, llama.cpp's from `timings`. The
engines run one after the other, never together, and each is started fresh.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


REQUEST_TIMEOUT = 300


def post(port, path, body):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.read().decode()


def note(engine, words, row):
    print(f'  {engine} {words} words: {row}', file=sys.stderr, flush=True)


def wait_ready(process, port, path):
    for _ in range(1200):
        if process.poll() is not None:
            raise SystemExit(f'server exited with {process.returncode}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise SystemExit('server did not become ready')


def prompt_of(words):
    return 'Summarise this list in one sentence: ' + ' '.join(f'item{i}' for i in range(words))


def run_allpaka(binary, model, contexts, new_tokens):
    port = free_port()
    log = open(os.devnull, 'w')
    server = subprocess.Popen([binary, 'serve', '--model', model, '--bind', f'127.0.0.1:{port}'],
                              stdout=log, stderr=log, env=dict(os.environ, ALLPAKA_RAG_TOOLS='0'))
    rows = []
    try:
        wait_ready(server, port, '/v1/models')
        for words in contexts:
            # Streaming carries the timing block on its last event.
            try:
                text = post(port, '/v1/chat/completions', {
                    'stream': True, 'temperature': 0, 'repeat_penalty': 1.0, 'max_tokens': new_tokens,
                    'messages': [{'role': 'user', 'content': prompt_of(words)}]})
            except OSError as error:
                # A request that outlives the timeout is a result, not a crash;
                # the server is still busy with it, so stop here.
                rows.append({'prompt_tokens': None, 'new_tokens': 0, 'decode_tok_s': 0.0,
                             'prefill_s': None, 'error': repr(error)[:60]})
                note('allpaka', words, rows[-1])
                break
            events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
            last = next(e for e in reversed(events) if 'timing' in e)
            rows.append({'prompt_tokens': last['usage']['prompt_tokens'],
                         'new_tokens': last['usage']['completion_tokens'],
                         'decode_tok_s': round(last['timing']['tokens_per_sec'], 1),
                         'prefill_s': round(last['timing']['prefill_secs'], 2)})
            note('allpaka', words, rows[-1])
    finally:
        server.kill()
        server.wait()
    return rows


def run_llama(binary, model, contexts, new_tokens):
    port = free_port()
    log = open(os.devnull, 'w')
    server = subprocess.Popen([binary, '-m', model, '--port', str(port), '-ngl', '99', '-c', '40960'],
                              stdout=log, stderr=log)
    rows = []
    try:
        wait_ready(server, port, '/health')
        for words in contexts:
            reply = json.loads(post(port, '/v1/chat/completions', {
                'temperature': 0, 'max_tokens': new_tokens, 'cache_prompt': False,
                'messages': [{'role': 'user', 'content': prompt_of(words)}]}))
            timings = reply['timings']
            rows.append({'prompt_tokens': reply['usage']['prompt_tokens'],
                         'new_tokens': reply['usage']['completion_tokens'],
                         'decode_tok_s': round(timings['predicted_per_second'], 1),
                         'prefill_s': round(timings['prompt_ms'] / 1000, 2)})
            note('llama', words, rows[-1])
    finally:
        server.kill()
        server.wait()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--allpaka', required=True)
    p.add_argument('--llama-server', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--contexts', type=int, nargs='+', default=[500, 2000, 4000, 6000],
                   help='prompt sizes in filler words; a word is about two tokens')
    p.add_argument('--new-tokens', type=int, default=64)
    p.add_argument('--json')
    a = p.parse_args()
    ours = run_allpaka(a.allpaka, a.model, a.contexts, a.new_tokens)
    theirs = run_llama(a.llama_server, a.model, a.contexts, a.new_tokens)
    print(f"{'words':>6} | {'allpaka ctx':>11} {'tok/s':>7} {'prefill s':>9} | {'llama ctx':>9} {'tok/s':>7} {'prefill s':>9} | ratio")
    missing = {'prompt_tokens': None, 'decode_tok_s': 0.0, 'prefill_s': None}
    for i, (words, t) in enumerate(zip(a.contexts, theirs)):
        o = ours[i] if i < len(ours) else missing
        print(f"{words:>6} | {str(o['prompt_tokens']):>11} {o['decode_tok_s']:>7} {str(o['prefill_s']):>9} | "
              f"{t['prompt_tokens']:>9} {t['decode_tok_s']:>7} {t['prefill_s']:>9} | "
              f"{o['decode_tok_s'] / t['decode_tok_s'] * 100:.0f}%")
    if a.json:
        with open(a.json, 'w') as out:
            json.dump({'contexts_words': a.contexts, 'new_tokens': a.new_tokens,
                       'allpaka': ours, 'llama': theirs}, out, indent=2)
            out.write('\n')


if __name__ == '__main__':
    main()
