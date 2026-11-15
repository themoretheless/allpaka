#!/usr/bin/env python3
"""allpaka against llama.cpp on one exact token stream.

    scripts/bench-token-replay.py --allpaka <allpaka binary> --llama-server <llama-server binary>
        --model <model.gguf> --out <report.json> [--prompt-file text.txt]
        [--new-tokens 128] [--rounds 3]

The prompt is tokenized once, by llama-server, and the same ids go to both
engines: `allpaka bench --engine` reads them from ALLPAKA_BENCH_TOKENS, and
llama-server gets them as a token array on /completion. Both prefill the
stream into an empty cache and decode greedily from its end. Rounds alternate
the engines; each engine runs in its own fresh process per round.

The report (schema `allpaka.token-replay/1`) records the prompt ids, each
engine's rates and decoded tokens per round, and a comparability verdict:

  prompt_identical   the same ids were prefilled by both (by construction)
  decode_identical   both decoded the same tokens; if not, the first
                     position where they part ways
  comparable         prompt identical and decode identical, so prefill and
                     decode rates measure the same work

For MoE models the same tokens do not mean the same expert routing: that is
recorded as a caveat, not checked.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import tempfile
import time
import urllib.request

SCHEMA = 'allpaka.token-replay/1'
DEFAULT_PROMPT = (
    'You are reviewing a pull request that changes how a key-value cache is allocated on a GPU. '
    'Explain, step by step, what could go wrong when a session is rebuilt in place with a larger '
    'capacity, which state might still refer to the old buffer, and how you would test for it. '
) * 12


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def post(port, path, body, timeout=600):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class LlamaServer:
    def __init__(self, binary, model):
        self.port = free_port()
        self.process = subprocess.Popen([binary, '-m', model, '--port', str(self.port), '-ngl', '99',
                                         '-c', '40960'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(1200):
            if self.process.poll() is not None:
                raise SystemExit('llama-server exited')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                time.sleep(0.5)
        raise SystemExit('llama-server did not become ready')

    def tokenize(self, text):
        return [t if isinstance(t, int) else t['id'] for t in
                post(self.port, '/tokenize', {'content': text, 'add_special': True})['tokens']]

    def completion(self, ids, n):
        reply = post(self.port, '/completion', {
            'prompt': ids, 'n_predict': n, 'temperature': 0, 'top_k': 1, 'cache_prompt': False,
            'return_tokens': True, 'ignore_eos': True})
        timings = reply['timings']
        return {'prefill_tok_s': round(timings['prompt_per_second'], 1),
                'decode_tok_s': round(timings['predicted_per_second'], 1),
                'tokens': reply.get('tokens', [])}

    def close(self):
        self.process.kill()
        self.process.wait()


def run_allpaka(binary, model, ids_path, n, report_path):
    env = dict(os.environ, ALLPAKA_BENCH_TOKENS=ids_path, ALLPAKA_BENCH_TG=str(n),
               ALLPAKA_BENCH_REPORT=report_path, ALLPAKA_BENCH_SKIP_MTP='1')
    done = subprocess.run([binary, 'bench', '--engine', model], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f'allpaka bench failed:\n{done.stderr[-2000:]}')
    with open(report_path) as f:
        report = json.load(f)
    by_name = {m['name']: m for m in report['measurements']}
    decode = by_name['decode']
    # Decode inputs are the first token and then each generated one but the
    # last; the generated stream is the inputs shifted by one.
    inputs = decode['input_tokens']
    return {'prefill_tok_s': round(by_name['prefill']['summary']['median'], 1),
            'decode_tok_s': round(decode['summary']['median'], 1),
            'tokens': inputs, 'prompt_tokens': by_name['prefill']['input_tokens']}


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--allpaka', required=True)
    p.add_argument('--llama-server', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--prompt-file')
    p.add_argument('--new-tokens', type=int, default=128)
    p.add_argument('--rounds', type=int, default=3)
    a = p.parse_args()
    text = open(a.prompt_file, encoding='utf-8').read() if a.prompt_file else DEFAULT_PROMPT

    work = tempfile.mkdtemp(prefix='allpaka-replay-')
    llama = LlamaServer(a.llama_server, a.model)
    try:
        ids = llama.tokenize(text)
        llama.completion(ids, 8)  # warm-up, not recorded
    finally:
        llama.close()
    ids_path = os.path.join(work, 'prompt.json')
    with open(ids_path, 'w') as f:
        json.dump(ids, f)

    rounds = []
    for r in range(a.rounds):
        order = ['allpaka', 'llama'] if r % 2 == 0 else ['llama', 'allpaka']
        row = {}
        for engine in order:
            if engine == 'allpaka':
                row['allpaka'] = run_allpaka(a.allpaka, a.model, ids_path, a.new_tokens,
                                             os.path.join(work, f'allpaka-{r}.json'))
            else:
                server = LlamaServer(a.llama_server, a.model)
                try:
                    server.completion(ids, 8)
                    row['llama'] = server.completion(ids, a.new_tokens)
                finally:
                    server.close()
        rounds.append(row)
        print(f"round {r + 1}: allpaka pp {row['allpaka']['prefill_tok_s']} tg {row['allpaka']['decode_tok_s']}"
              f" | llama pp {row['llama']['prefill_tok_s']} tg {row['llama']['decode_tok_s']}", flush=True)

    ours, theirs = rounds[0]['allpaka'], rounds[0]['llama']
    prompt_identical = all(r['allpaka']['prompt_tokens'] == ids for r in rounds)
    split = first_difference(ours['tokens'], theirs['tokens'])
    stable = all(r['allpaka']['tokens'] == ours['tokens'] and r['llama']['tokens'] == theirs['tokens']
                 for r in rounds)
    median = lambda engine, key: statistics.median(r[engine][key] for r in rounds)
    report = {
        'schema': SCHEMA,
        'model': os.path.basename(a.model),
        'prompt_tokens': ids,
        'new_tokens': a.new_tokens,
        'rounds': [{engine: {k: v for k, v in row[engine].items() if k != 'prompt_tokens'}
                    for engine in row} for row in rounds],
        'median': {engine: {'prefill_tok_s': median(engine, 'prefill_tok_s'),
                            'decode_tok_s': median(engine, 'decode_tok_s')} for engine in ('allpaka', 'llama')},
        'comparability': {
            'prompt_identical': prompt_identical,
            'decode_identical': split is None,
            'decode_first_difference': split,
            'streams_stable_across_rounds': stable,
            'comparable': prompt_identical and split is None,
            'caveats': ['For MoE models identical tokens do not imply identical expert routing.'],
        },
    }
    with open(a.out, 'w') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    c = report['comparability']
    print(f"prompt identical: {c['prompt_identical']}; decode identical: {c['decode_identical']}"
          + ('' if split is None else f' (first difference at token {split})')
          + f"; comparable: {c['comparable']}")
    m = report['median']
    print(f"median prefill {m['allpaka']['prefill_tok_s']} vs {m['llama']['prefill_tok_s']} "
          f"({m['allpaka']['prefill_tok_s'] / m['llama']['prefill_tok_s'] * 100:.0f}%), "
          f"decode {m['allpaka']['decode_tok_s']} vs {m['llama']['decode_tok_s']} "
          f"({m['allpaka']['decode_tok_s'] / m['llama']['decode_tok_s'] * 100:.0f}%)")


if __name__ == '__main__':
    main()
