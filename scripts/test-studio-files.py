#!/usr/bin/env python3
"""File-browser HTTP contract on a temporary Studio; never calls a model."""
import json,os,pathlib,socket,subprocess,tempfile,time,urllib.request,urllib.error
binary=pathlib.Path(os.environ.get('ALLPAKA_TEST_BINARY','target/debug/allpaka')).resolve()
with tempfile.TemporaryDirectory(prefix='studio-files-http-') as tmp:
 root=pathlib.Path(tmp); workspace=root/'project';workspace.mkdir();data=workspace/'history';data.mkdir()
 (workspace/'example.rs').write_text('first\nsecond\nthird\n');(workspace/'.env').write_text('SECRET');(data/'private').write_text('HISTORY_SECRET')
 (workspace/'binary').write_bytes(b'a\x00b');(workspace/'large').write_bytes(b'x'*100001);(workspace/'outside').symlink_to(root)
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();base=f'http://127.0.0.1:{port}'
 with (root/'server.log').open('w') as log:
  proc=subprocess.Popen([str(binary),'studio','--bind',f'127.0.0.1:{port}','--workspace',str(workspace),'--data-dir',str(data)],stdout=log,stderr=log)
  def api(path,body=None):
   req=urllib.request.Request(base+'/api/'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json','X-Allpaka-Client':'studio'})
   with urllib.request.urlopen(req,timeout=10) as r:return json.load(r)
  try:
   for i in range(100):
    try:api('config');break
    except (OSError,urllib.error.URLError):time.sleep(.1)
   else:raise AssertionError('Studio did not start')
   request={'project_id':'default','root':'workspace','path':'.'}
   rows=api('files/list',request)['entries'];names={f['name'] for f in rows};assert 'example.rs' in names and not names.intersection({'.env','history','outside'})
   found=api('files/search',dict(request,query='exrs'));assert [r['path'] for r in found['entries']]==['example.rs']
   result=api('files/read',dict(request,path='example.rs'));assert result['text']=='first\nsecond\nthird\n' and result['bytes']==19
   for patch in [{'path':'../project/example.rs'},{'path':'.env'},{'path':'history/private'},{'path':'outside'},{'path':'binary'},{'path':'large'},{'root':'unknown'},{'project_id':'missing'}]:
    try:api('files/read',dict(request,**patch));raise AssertionError('unsafe read allowed: '+str(patch))
    except urllib.error.HTTPError as e:assert e.code in (400,404)
   for asset in ['/', '/files.js']:
    with urllib.request.urlopen(base+asset) as r:assert r.status==200
   print('PASS files HTTP: search, preview, byte counts, root/history/hidden/symlink/binary/size boundaries and UI asset')
  finally:proc.terminate();proc.wait(timeout=10)
