#!/usr/bin/env python3
"""Attachment extraction HTTP contract on a temporary Studio; never calls a model."""
import hashlib,http.server,json,os,pathlib,socket,subprocess,tempfile,threading,time,urllib.parse,urllib.request,urllib.error
binary=pathlib.Path(os.environ.get('ALLPAKA_TEST_BINARY','target/debug/allpaka')).resolve()
PAGE=b'<!doctype html><html><head><style>x{}</style></head><body><h1>Title</h1><p>Body &amp; more</p></body></html>'
class Page(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  body=PAGE if self.path=='/page.html' else b'x'*(10<<20)
  self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def log_message(self,*a):pass
site=http.server.ThreadingHTTPServer(('127.0.0.1',0),Page);threading.Thread(target=site.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory(prefix='studio-extract-http-') as tmp:
 root=pathlib.Path(tmp); workspace=root/'project';workspace.mkdir();data=root/'data';data.mkdir()
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();base=f'http://127.0.0.1:{port}'
 with (root/'server.log').open('w') as log:
  proc=subprocess.Popen([str(binary),'studio','--bind',f'127.0.0.1:{port}','--workspace',str(workspace),'--data-dir',str(data)],stdout=log,stderr=log,env=dict(os.environ,ALLPAKA_TESSERACT='allpaka-no-such-tesseract'))
  def call(path,body=None,raw=None):
   data_=raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
   req=urllib.request.Request(base+'/api/'+path,data=data_,headers={'Content-Type':'application/octet-stream' if raw is not None else 'application/json','X-Allpaka-Client':'studio'})
   with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
  def rejected(path,body=None,raw=None):
   try:call(path,body,raw)
   except urllib.error.HTTPError as e:return e.code,json.load(e)['error']
   raise AssertionError('accepted: '+path)
  try:
   for i in range(100):
    try:call('config');break
    except (OSError,urllib.error.URLError):time.sleep(.1)
   else:raise AssertionError('Studio did not start')
   note='Привет, notes\n'.encode()
   r=call('extract/upload?name=notes.txt',raw=note)
   assert r['kind']=='text' and r['text']=='Привет, notes\n' and r['sha256']==hashlib.sha256(note).hexdigest() and not r['truncated']
   assert (data/r['original']).read_bytes()==note and r['attachment'].startswith('<attachment name="notes.txt"')
   assert call('extract/'+r['id'])['sha256']==r['sha256']
   code,msg=rejected('extract/upload?name=x.bin',raw=b'\x7fELF\x02\x01\x01\x00\xff\xfe');assert code==400 and 'Unsupported' in msg,msg
   code,msg=rejected('extract/upload?name=a.pdf',raw=note);assert code==400 and 'says Pdf' in msg,msg
   code,msg=rejected('extract/upload?name=big.txt',raw=b'a'*(2*1024*1024+1));assert code==400 and 'limit' in msg,msg
   code,msg=rejected('extract/upload?name=scan.png',raw=b'\x89PNG\r\n\x1a\n....');assert code==400 and 'allpaka-no-such-tesseract' in msg,msg
   page=f'http://127.0.0.1:{site.server_port}/page.html'
   r=call('extract/url',{'url':page});assert r['kind']=='html' and r['text']=='Title\n\nBody & more' and r['url']==page,r
   assert (data/r['original']).read_bytes()==PAGE
   code,msg=rejected('extract/url',{'url':f'http://127.0.0.1:{site.server_port}/huge'});assert code==400 and 'larger' in msg,msg
   code,msg=rejected('extract/url',{'url':'file:///etc/passwd'});assert code==400,msg
   code,_=rejected('extract/missing');assert code==404
   print('PASS extract HTTP: upload/url extraction, magic-byte typing, size caps, missing-tool error, preserved originals')
  finally:proc.terminate();proc.wait(timeout=10);site.shutdown()
