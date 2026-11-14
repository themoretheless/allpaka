#!/usr/bin/env python3
"""Two long chat requests at once: does the second wait for all of the first?

    scripts/bench-two-long.py --addr 127.0.0.1:8080 [--prompt-tokens 4000]
        [--new-tokens 512] [--json out.json]

Starts request A, waits until it streams, then starts request B (same size).
Reports each one's time to first token, median gap between tokens and total
time. A server that runs requests one after another gives B a first token
after about all of A; one that interleaves gives it within a few steps, with
both decoding at about half rate while they overlap.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def stream(addr, prompt_tokens, new_tokens, tag, started, out):
    filler = ' '.join(f'{tag}{i}' for i in range(prompt_tokens // 2))
    body = {'stream': True, 'temperature': 0, 'max_tokens': new_tokens,
            'messages': [{'role': 'user', 'content':
                          'Write a long story about the following words, never stopping:\n\n' + filler}]}
    request = urllib.request.Request(f'http://{addr}/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    sent = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for line in response:
            if line.startswith(b'data: {'):
                if json.loads(line[6:])['choices'][0].get('delta', {}).get('content'):
                    arrivals.append(time.perf_counter())
                    started.set()
    started.set()
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    out[tag] = {
        'first_token_ms': round((arrivals[0] - sent) * 1000, 1) if arrivals else None,
        'median_gap_ms': round(statistics.median(gaps) * 1000, 2) if gaps else None,
        'deltas': len(arrivals),
        'total_ms': round((time.perf_counter() - sent) * 1000, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--addr', default='127.0.0.1:8080')
    p.add_argument('--prompt-tokens', type=int, default=4000)
    p.add_argument('--new-tokens', type=int, default=512)
    p.add_argument('--json')
    a = p.parse_args()
    out = {}
    a_started, b_started = threading.Event(), threading.Event()
    ta = threading.Thread(target=stream, args=(a.addr, a.prompt_tokens, a.new_tokens, 'a', a_started, out))
    tb = threading.Thread(target=stream, args=(a.addr, a.prompt_tokens, a.new_tokens, 'b', b_started, out))
    ta.start()
    a_started.wait(timeout=600)
    tb.start()
    ta.join()
    tb.join()
    for tag in 'ab':
        r = out[tag]
        print(f"{tag}: first token {r['first_token_ms']} ms, median gap {r['median_gap_ms']} ms, "
              f"{r['deltas']} deltas, total {r['total_ms']} ms")
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'prompt_tokens': a.prompt_tokens, 'new_tokens': a.new_tokens, **out}, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
