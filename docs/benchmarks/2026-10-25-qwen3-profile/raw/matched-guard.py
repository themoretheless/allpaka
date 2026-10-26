import subprocess,time,json,os,re,plistlib,threading,itertools,statistics,gzip,hashlib
from pathlib import Path
ROOT=Path('/tmp/allpaka-profile-20261025'); OUT=ROOT/'matched-guarded';OUT.mkdir(exist_ok=True)
MODEL='/Users/themoretheless/.cache/allpaka/models/Qwen3-30B-A3B-Q4_K_M.gguf'
BINS={'before':str(ROOT/'forced-before'),'after':str(ROOT/'forced-after'),'llama':str(ROOT/'llama-matched')}
# An idle desktop still has compositor/GUI activity: record it and use explicit
# admission limits instead of claiming absolute OS/GPU exclusivity.
def snapshot():
 cpu={}
 for l in subprocess.check_output(['ps','-Ao','pid=,time=,comm='],text=True).splitlines():
  p,t,c=l.strip().split(None,2); fields=t.split(':'); secs=sum(float(x)*60**i for i,x in enumerate(reversed(fields)));cpu[int(p)]=(secs,c)
 gpu={}
 def walk(n):
  m=re.match(r'pid (\d+), (.*)',n.get('IOUserClientCreator',''))
  if m:
   k=int(m[1]);gpu[k]=gpu.get(k,0)+sum(x.get('accumulatedGPUTime',0) for x in n.get('AppUsage',[]) or [])
  for c in n.get('IORegistryEntryChildren',[]):walk(c)
 for n in plistlib.loads(subprocess.check_output(['ioreg','-a','-l','-r','-c','AGXAccelerator'])):walk(n)
 return time.monotonic(),cpu,gpu

def diff(a,b,allowed):
 dt=b[0]-a[0]; cpu=[];gpu=[]
 for pid,(t,c) in b[1].items():
  if pid in allowed or pid not in a[1]:continue
  pct=max(0,t-a[1][pid][0])/dt*100
  if pct>2:cpu.append({'pid':pid,'command':c,'percent':round(pct,2)})
 for pid,t in b[2].items():
  if pid in allowed or pid not in a[2]:continue
  pct=max(0,t-a[2][pid])/dt/1e9*100
  if pct>.1:gpu.append({'pid':pid,'command':b[1].get(pid,(0,'gone'))[1],'percent':round(pct,3)})
 blockers=[x for x in cpu if re.search(r'/(rustc|cargo|clang|clang\+\+|ninja|make|xctrace|street_walk)$',x['command']) and x['percent']>5]
 # No active compiler/renderer; <= one CPU core of background work and
 # <=2% background GPU excluding the unavoidable WindowServer compositor.
 total=sum(x['percent'] for x in cpu);g=sum(x['percent'] for x in gpu if not x['command'].endswith('/WindowServer'))
 return {'observed_unix':time.time(),'background_cpu_percent':round(total,2),'background_gpu_percent_excluding_compositor':round(g,3),'cpu':cpu,'gpu':gpu,'quiet':not blockers and total<=100 and g<=2}

accepted=[];attempts=[];reference=None;refgen=None; deadline=time.monotonic()+240
for attempt in range(1,25):
 if len(accepted)>=6 or time.monotonic()>deadline:break
 # Require three consecutive quiet 2-second samples before launching a block.
 quiet=0
 while quiet<3 and time.monotonic()<deadline:
  a=snapshot();time.sleep(2);d=diff(a,snapshot(),{os.getpid()})
  quiet=quiet+1 if d['quiet'] else 0
  with (OUT/'admission.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
 if quiet<3:break
 block={'attempt':attempt,'order':list(itertools.permutations(BINS))[len(accepted)%6], 'samples':{},'load':[]}
 print('attempt',attempt,'accepted',len(accepted),'order',block['order'],flush=True)
 for arm in block['order']:
  commands=[('both',[BINS[arm],MODEL,'480','3']+([] if arm=='llama' else ['forced']))]
  block['samples'][arm]={}
  for phase,cmd in commands:
   stem=OUT/f'attempt{attempt}-{arm}-{phase}';a=snapshot()
   with stem.with_suffix('.stdout').open('w') as out,stem.with_suffix('.log').open('w') as err:
    p=subprocess.Popen(cmd,stdout=out,stderr=err);allowed={os.getpid(),p.pid}
    while p.poll() is None:
     time.sleep(1);b=snapshot();block['load'].append(diff(a,b,allowed));a=b
   assert p.returncode==0,(arm,p.returncode)
   text=stem.with_suffix('.stdout').read_text()
   r=json.loads(next(x for x in reversed(text.splitlines()) if x.startswith('{')))
   assert r['decode_kind']=='forced_full_logits' and r['kv_format']=='f16'
   assert r['forced_decode_inputs']==[((i+32)*733+17)%30000 for i in range(32)]
   prefill_logits=r.pop('prefill_logits');decode_logits=r.pop('decode_final_logits')
   if arm!='llama':
    if reference is None:reference=(prefill_logits,decode_logits)
    assert all(x['successes']==32 and x['declines']==0 for x in r['decode_fast_path'])
    for k,got,ref in zip(['prefill','decode'],[prefill_logits,decode_logits],reference):
     delta=max(abs(x-y) for x,y in zip(got,ref));assert len(got)==len(ref) and delta<1e-4
     r[k+'_max_logit_difference']=delta
   else:
    assert 'offloaded 49/49 layers to GPU' in stem.with_suffix('.log').read_text()
    for k,got,ref in zip(['prefill','decode'],[prefill_logits,decode_logits],reference):
     assert len(got)==len(ref)
     r[k+'_max_cross_engine_logit_difference']=max(abs(x-y) for x,y in zip(got,ref))
     r[k+'_cross_engine_argmax_match']=max(range(len(got)),key=got.__getitem__)==max(range(len(ref)),key=ref.__getitem__)
   r['prefill_logit_sha256']=hashlib.sha256(json.dumps(prefill_logits).encode()).hexdigest()
   r['decode_logit_sha256']=hashlib.sha256(json.dumps(decode_logits).encode()).hexdigest()
   stem.with_suffix('.json').write_text(json.dumps(r,indent=2))
   for k in ['prefill','decode']:
    block['samples'][arm][k]=r[k+'_tokens_per_second']
    assert len(block['samples'][arm][k])==3 and all(v>0 for v in block['samples'][arm][k])
   with gzip.open(stem.with_suffix('.stdout.gz'),'wt') as f:f.write(text)
   stem.with_suffix('.stdout').unlink()
   stem.with_suffix('.command.json').write_text(json.dumps({'argv':cmd,'exit_code':p.returncode}))
  print(arm,{k:round(statistics.median(v),2) for k,v in block['samples'][arm].items()},flush=True)
 block['accepted']=all(x['quiet'] for x in block['load']);attempts.append(block)
 if block['accepted']:accepted.append(block)
 print('accepted=',block['accepted'],flush=True)
 (OUT/'attempts.json').write_text(json.dumps(attempts,indent=2));(OUT/'accepted.json').write_text(json.dumps(accepted,indent=2))
(OUT/'status.json').write_text(json.dumps({'accepted':len(accepted),'required':6,'complete':len(accepted)==6,'binary_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in BINS.items()},'model':MODEL},indent=2))
