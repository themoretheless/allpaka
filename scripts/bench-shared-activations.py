#!/usr/bin/env python3
"""Sequential A/B of one frozen release binary, including output parity checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary', type=Path)
p.add_argument('model', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--pairs', type=int, default=7)
p.add_argument('--rounds', type=int, default=3)
p.add_argument('--pp', type=int, nargs='+', default=[512, 2048])
a = p.parse_args()
assert a.pairs > 1 and a.rounds > 0 and all(n > 0 for n in a.pp)
a.output.mkdir(parents=True, exist_ok=False)
binary, model = a.binary.resolve(), a.model.resolve()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()
metadata = {'binary': str(binary), 'binary_sha256': sha(binary),
            'model': str(model), 'model_sha256': sha(model),
            'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
            'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD', '--', 'crates/allpaka-backend', 'crates/allpaka-model'])).hexdigest(),
            'pairs': a.pairs, 'rounds': a.rounds, 'pp': a.pp,
            'protocol': 'alternating order per pair; fresh process; one warmup round; fixed tokens; same binary',
            'limitations': ['Other desktop applications may use CPU/GPU; pilot excluded',
                            'Single M4 Max and one Q8_0 model; not M1/M2/M3 qualification']}
(a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
summary = {}
for pp in a.pp:
    pairs = []
    reference = None
    for pair in range(a.pairs):
        result = {}
        for enabled in ([0, 1] if pair % 2 == 0 else [1, 0]):
            stem = f'pp{pp}-pair{pair+1}-mode{enabled}'
            env = os.environ.copy()
            env['ALLPAKA_PREFILL_ZERO_COPY'] = str(enabled)
            start = time.time()
            run = subprocess.run([str(binary), str(model), str(pp), str(a.rounds)],
                                 env=env, capture_output=True, text=True, check=True)
            (a.output / f'{stem}.log').write_text(run.stderr)
            record = json.loads(next(line for line in reversed(run.stdout.splitlines()) if line.startswith('{')))
            logits = record.pop('prefill_logits')
            if reference is None:
                reference = logits
            assert len(logits) == len(reference)
            max_diff = max(abs(x-y) for x,y in zip(logits, reference))
            assert max_diff < 1e-4, (stem, max_diff)
            record['max_logit_diff_vs_reference'] = max_diff
            record['logit_sha256'] = hashlib.sha256(json.dumps(logits, separators=(',', ':')).encode()).hexdigest()
            record['started_unix'] = start
            assert record['gpu_dispatches'] > 0
            assert (record['shared_outputs'] > 0) == bool(enabled)
            assert (record['projection_bytes_avoided'] > 0) == bool(enabled)
            (a.output / f'{stem}.json').write_text(json.dumps(record, indent=2) + '\n')
            result[enabled] = record
        assert result[0]['generated'] == result[1]['generated']
        pairs.append(result)
        print(f'PP{pp} pair {pair+1}/{a.pairs}: '
              + ' / '.join(f"{statistics.median(result[k]['prefill_tokens_per_second']):.1f}" for k in [0,1]), flush=True)
    summary[str(pp)] = {}
    for phase in ['prefill', 'decode']:
        key = phase + '_tokens_per_second'
        medians = {mode: [statistics.median(pair[mode][key]) for pair in pairs] for mode in [0,1]}
        ratios = [b / a for a,b in zip(medians[0], medians[1])]
        rng = random.Random(42)
        bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
        summary[str(pp)][phase] = {
            'copy_pair_medians': medians[0], 'shared_pair_medians': medians[1],
            'copy_median': statistics.median(medians[0]), 'shared_median': statistics.median(medians[1]),
            'paired_change_percent': (statistics.median(ratios)-1)*100,
            'paired_bootstrap_95_percent': [(bootstrap[i]-1)*100 for i in [250,9749]],
        }
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
