#!/usr/bin/env python3
"""Balanced three-arm Metal comparison; same GGUF, PP/TG boundaries, f16 KV."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('allpaka', type=Path, help='shared_activation_bench release example')
p.add_argument('llama', type=Path)
p.add_argument('model', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--pp', type=int, nargs='+', default=[480, 2048])
p.add_argument('--blocks', type=int, default=6)
p.add_argument('--order-offset', type=int, default=0)
p.add_argument('--rounds', type=int, default=3)
p.add_argument('--threads', type=int, default=os.cpu_count())
a = p.parse_args()
assert a.blocks >= 1 and a.rounds >= 1 and a.threads > 0 and all(0 < n <= 32768 for n in a.pp)
a.output.mkdir(parents=True, exist_ok=False)
ap, llama, model = (v.resolve() for v in [a.allpaka, a.llama, a.model])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(16 << 20): h.update(data)
    return h.hexdigest()

def run(cmd, stem, env=None):
    started = time.time()
    r = subprocess.run(cmd, text=True, capture_output=True, env=env)
    (a.output / (stem + '.stdout')).write_text(r.stdout)
    (a.output / (stem + '.log')).write_text(r.stderr)
    (a.output / (stem + '.command.json')).write_text(json.dumps({'argv': cmd, 'exit_code': r.returncode, 'started_unix': started}, indent=2)+'\n')
    r.check_returncode()
    return r.stdout

version = subprocess.run([str(llama), '--version'], text=True, capture_output=True)
(a.output / 'llama-version.txt').write_text(version.stdout + version.stderr)
meta = {'allpaka_binary': str(ap), 'allpaka_sha256': sha(ap), 'llama_binary': str(llama),
        'llama_sha256': sha(llama), 'model': str(model), 'model_sha256': sha(model),
        'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()),
        'pp': a.pp, 'tg': 32, 'decode_context_tokens': 1, 'kv': 'f16', 'chunk_and_ubatch': 512,
        'llama_batch': 512, 'llama_flash_attention': 'on', 'threads': a.threads,
        'blocks': a.blocks, 'order_offset': a.order_offset, 'measured_rounds_per_process': a.rounds,
        'allpaka_warmup_rounds': 1, 'comparison_validated': False,
        'limitations': ['llama-bench uses its own token stream; MoE expert routing differs',
                        'Desktop background load is uncontrolled; report paired uncertainty',
                        'Fresh model hash and llama version; do not compare numerically to historical reports']}
(a.output / 'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
orders = list(itertools.permutations(['copy','shared','llama']))
summary = {}
for pp in a.pp:
    reference = None
    reference_tokens = None
    blocks = []
    for block in range(a.blocks):
        results = {}
        order = orders[(block + a.order_offset) % len(orders)]
        print(f'PP{pp} block {block+1}/{a.blocks}: {" -> ".join(order)}',flush=True)
        for arm in order:
            stem = f'pp{pp}-block{block+1}-{arm}'
            if arm in ['copy','shared']:
                env = os.environ.copy()
                env['ALLPAKA_PREFILL_ZERO_COPY'] = '1' if arm=='shared' else '0'
                text = run([str(ap),str(model),str(pp),str(a.rounds),'greedy'],stem,env)
                s = json.loads(next(line for line in reversed(text.splitlines()) if line.startswith('{')))
                logits = s.pop('prefill_logits')
                if reference is None: reference = logits
                assert len(logits)==len(reference)
                s['max_logit_difference'] = max(abs(x-y) for x,y in zip(logits,reference))
                assert s['max_logit_difference'] < 1e-4
                s['logit_sha256'] = hashlib.sha256(json.dumps(logits,separators=(',',':')).encode()).hexdigest()
                if reference_tokens is None: reference_tokens = s['generated']
                assert s['generated']==reference_tokens
                assert all(len(s[k])==a.rounds and all(math.isfinite(x) and x>0 for x in s[k]) for k in ['prefill_tokens_per_second','decode_tokens_per_second'])
                assert s['decode_kind']=='forward_greedy_n' and s['decode_context_tokens']==1
                assert s['kv_format']=='f16' and s['decode_tokens']==32
                assert all(r['attempts']==32 and r['successes']==32 and r['declines']==0 for r in s['decode_fast_path'])
                assert (s['shared_outputs']>0)==(arm=='shared')
                assert (s['projection_bytes_avoided']>0)==(arm=='shared')
                results[arm] = {'prefill':s['prefill_tokens_per_second'],'decode':s['decode_tokens_per_second'],'generated':s['generated']}
                (a.output / (stem+'.json')).write_text(json.dumps(s,indent=2)+'\n')
                # The normalized JSON retains checksums/parity; raw large logits
                # stay in a compressed artifact instead of bloating the report.
                import gzip
                with gzip.open(a.output/(stem+'.stdout.gz'),'wt') as f:f.write(text)
                (a.output/(stem+'.stdout')).unlink()
            else:
                results[arm] = {}
                for phase,prompt,gen,depth in [('prefill',pp,0,0),('decode',0,32,1)]:
                    cmd=[str(llama),'-m',str(model),'-p',str(prompt),'-n',str(gen),'-d',str(depth),
                         '-r',str(a.rounds),'-ngl','99','-ctk','f16','-ctv','f16',
                         '-b','512','-ub','512','-fa','on','-t',str(a.threads),'-o','json']
                    rows=json.loads(run(cmd,stem+'-'+phase))
                    matches=[r for r in rows if r['n_prompt']==prompt and r['n_gen']==gen and r['n_depth']==depth]
                    assert len(matches)==1
                    r=matches[0]
                    assert 'MTL' in r['backends'] and r['n_gpu_layers']==99 and r['n_cpu_moe']==0
                    assert r['type_k']==r['type_v']=='f16' and not r['no_kv_offload']
                    assert len(r['samples_ts'])==a.rounds and all(math.isfinite(x) and x>0 for x in r['samples_ts'])
                    results[arm][phase]=r['samples_ts']
                    (a.output/(stem+'-'+phase+'.json')).write_text(json.dumps(r,indent=2)+'\n')
            print('  '+arm+': '+', '.join(f'{phase}={statistics.median(results[arm][phase]):.1f}' for phase in ['prefill','decode']),flush=True)
        assert results['copy']['generated']==results['shared']['generated']
        blocks.append(results)
    summary[str(pp)]={}
    for phase in ['prefill','decode']:
        medians={arm:[statistics.median(b[arm][phase]) for b in blocks] for arm in ['copy','shared','llama']}
        rng=random.Random(42)
        def ratios(before,after):
            values=[y/x for x,y in zip(medians[before],medians[after])]
            bootstrap=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
            return {'median_percent':100*(statistics.median(values)-1),
                    'bootstrap_95_percent':[100*(bootstrap[i]-1) for i in [250,9749]]}
        summary[str(pp)][phase]={
            'process_medians':medians,
            'median':{arm:statistics.median(v) for arm,v in medians.items()},
            'zero_copy_vs_copy':ratios('copy','shared'),
            'copy_vs_llama':ratios('llama','copy'),
            'shared_vs_llama':ratios('llama','shared')}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert sha(ap)==meta['allpaka_sha256'] and sha(llama)==meta['llama_sha256'], 'benchmark binary changed during run'
print(json.dumps(summary,indent=2),flush=True)
