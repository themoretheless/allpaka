import json,os,pathlib,random,signal,statistics,subprocess,sys,time
root=pathlib.Path('/tmp/allpaka-qwen30-zero-copy');out=root/'guarded';out.mkdir(exist_ok=True)
model=str(pathlib.Path.home()/'.cache/allpaka/models/Qwen3-30B-A3B-Q4_K_M.gguf')
blocks=[];manifest=[]
def renderers():
 return [r.strip() for r in subprocess.check_output(['ps','-Ao','pid,comm'],text=True).splitlines() if r.rstrip().endswith('/street_walk')]
def extract(path,prefix):
 b={}
 for arm in ['copy','shared']:
  s=json.loads((path/(prefix+arm+'.json')).read_text());b[arm]={'prefill':s['prefill_tokens_per_second'],'decode':s['decode_tokens_per_second'],'generated':s['generated'],'logit_sha256':s['logit_sha256']}
 for phase in ['prefill','decode']:
  s=json.loads((path/(prefix+'llama-'+phase+'.json')).read_text());b.setdefault('llama',{})[phase]=s['samples_ts']
 assert b['copy']['generated']==b['shared']['generated']
 assert b['copy']['logit_sha256']==b['shared']['logit_sha256']
 return b
# These two complete blocks ended before independently observed renderer start
# (06:00:23 local time); their command/end timestamps remain in the raw files.
for i in [1,2]:
 blocks.append(extract(root/'clean',f'pp480-block{i}-'))
 manifest.append({'order_offset':i-1,'source':str(root/'clean'),'prefix':f'pp480-block{i}-','criterion':'all arms completed before renderer PID 8914 started at 06:00:23'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.monotonic();attempt=0;child=None
try:
 while len(blocks)<6:
  if time.monotonic()-start>1800:raise TimeoutError('no sufficiently long renderer-free interval')
  while renderers():
   print('waiting for renderer to finish; retained',len(blocks),'blocks',flush=True);time.sleep(10)
   if time.monotonic()-start>1800:raise TimeoutError('renderer remains active')
  time.sleep(3)
  if renderers():continue
  attempt+=1;idx=len(blocks);dest=out/f'attempt-{attempt}-order{idx}'
  cmd=[sys.executable,'scripts/bench-zero-copy-vs-llama.py',str(root/'allpaka-shared-bench'),'/opt/homebrew/bin/llama-bench',model,str(dest),'--pp','480','--blocks','1','--order-offset',str(idx),'--rounds','3']
  log=open(out/f'attempt-{attempt}.log','w')
  print('starting renderer-free block',idx+1,'attempt',attempt,flush=True)
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  overlap=[]
  while child.poll() is None:
   found=renderers()
   if found:
    overlap=found;os.killpg(child.pid,signal.SIGTERM);child.wait();break
   time.sleep(0.5)
  log.close()
  if overlap:
   (out/f'attempt-{attempt}-excluded.json').write_text(json.dumps({'reason':'renderer appeared during block','renderers':overlap},indent=2)+'\n');print('excluded overlapping attempt',attempt,flush=True);continue
  if child.returncode!=0:raise RuntimeError(f'benchmark attempt {attempt} failed; inspect its log')
  b=extract(dest,'pp480-block1-')
  assert b['copy']['generated']==blocks[0]['copy']['generated'] and b['copy']['logit_sha256']==blocks[0]['copy']['logit_sha256']
  blocks.append(b);manifest.append({'order_offset':idx,'source':str(dest),'prefix':'pp480-block1-','criterion':'renderer absent throughout block; polled every 0.5 s'})
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('retained',len(blocks),'of 6 blocks',flush=True)
finally:
 if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()
summary={}
for phase in ['prefill','decode']:
 med={arm:[statistics.median(b[arm][phase]) for b in blocks] for arm in ['copy','shared','llama']}
 def compare(before,after):
  vals=[y/x for x,y in zip(med[before],med[after])];rng=random.Random(42)
  bs=sorted(statistics.median(rng.choices(vals,k=len(vals))) for _ in range(10000))
  return {'paired_percent':100*(statistics.median(vals)-1),'bootstrap_95_percent':[100*(bs[i]-1) for i in [250,9749]]}
 summary[phase]={'process_medians':med,'median':{k:statistics.median(v) for k,v in med.items()},'shared_vs_copy':compare('copy','shared'),'shared_vs_llama':compare('llama','shared'),'copy_vs_llama':compare('llama','copy')}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
(root/'guarded-complete').write_text('success\n')
