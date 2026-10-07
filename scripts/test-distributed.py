#!/usr/bin/env python3
"""Real-process regression: native GGUF -> two TCP stages -> chat gateway."""
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def fixture(path):
    def string(value):
        value = value.encode()
        return struct.pack('<Q', len(value)) + value
    metadata = []
    def meta(name, kind, value):
        metadata.append(string(name) + struct.pack('<I', kind) + value)
    meta('general.architecture', 8, string('llama'))
    for name, value in {'block_count':2, 'embedding_length':128, 'attention.head_count':2,
                        'attention.head_count_kv':1, 'attention.key_length':64,
                        'feed_forward_length':256}.items():
        meta('llama.' + name, 4, struct.pack('<I', value))
    meta('llama.attention.layer_norm_rms_epsilon', 6, struct.pack('<f', 1e-5))
    meta('llama.rope.freq_base', 6, struct.pack('<f', 10000))
    # GPT-2 byte alphabet, with one token per byte and two ChatML markers.
    visible = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    codes = dict(zip(visible, visible))
    extra = 256
    for byte in range(256):
        if byte not in codes:
            codes[byte] = extra
            extra += 1
    vocab = [chr(codes[i]) for i in range(256)] + ['<|im_start|>', '<|im_end|>']
    meta('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, len(vocab)) + b''.join(map(string, vocab)))
    meta('tokenizer.ggml.merges', 9, struct.pack('<IQ', 8, 0))
    tensors = []
    state = 42
    def matrix(count):
        nonlocal state
        values = []
        for _ in range(count):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64)-1)
            values.append(((state >> 40)/(1 << 24)-0.5)*0.2)
        return values
    def tensor(name, dims, values=None):
        size = 1
        for dim in dims: size *= dim
        tensors.append((name, dims, matrix(size) if values is None else values))
    tensor('token_embd.weight', [128,258])
    tensor('output_norm.weight', [128], [1.0]*128)
    tensor('output.weight', [128,258])
    for layer in range(2):
        for name, dims in [('attn_norm',[128]),('attn_q',[128,128]),('attn_k',[128,64]),
                           ('attn_v',[128,64]),('attn_output',[128,128]),('ffn_norm',[128]),
                           ('ffn_gate',[128,256]),('ffn_up',[128,256]),('ffn_down',[256,128])]:
            tensor(f'blk.{layer}.{name}.weight', dims, [1.0]*128 if name.endswith('norm') else None)
    infos, data = bytearray(), bytearray()
    for name, dims, values in tensors:
        data.extend(b'\0' * (-len(data) % 32))
        infos.extend(string(name) + struct.pack('<I',len(dims)) + struct.pack('<'+'Q'*len(dims),*dims) + struct.pack('<IQ',0,len(data)))
        data.extend(struct.pack('<'+'f'*len(values), *values))
    header = b'GGUF' + struct.pack('<IQQ',3,len(tensors),len(metadata)) + b''.join(metadata) + infos
    path.write_bytes(header + b'\0' * (-len(header) % 32) + data)


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def call(address, body=None, route='/v1/chat/completions'):
    request = urllib.request.Request(f'http://127.0.0.1:{address}{route}',
        data=None if body is None else json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    try: response = urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as error: response = error
    with response: return response.status, response.read().decode()


def main():
    binary = str(Path(os.environ.get('ALLPAKA_BINARY','target/debug/allpaka')).resolve())
    env = dict(os.environ, ALLPAKA_CLUSTER_KEY='regression-key-123456789', ALLPAKA_RAG_TOOLS='0')
    if not os.environ.get('ALLPAKA_TEST_GPU'): env['ALLPAKA_NO_GPU']='1'
    processes = []
    with tempfile.TemporaryDirectory(prefix='allpaka-distributed-') as folder:
        root = Path(folder)
        model = root/'fixture.gguf'
        fixture(model)
        first, second, gateway, local = [port() for _ in range(4)]
        def start(name, args):
            log = open(root/(name+'.log'), 'w')
            process = subprocess.Popen([binary]+args, env=dict(env, ALLPAKA_NO_GPU='1') if name == 'local' else env, stdout=log, stderr=log)
            processes.append((process,log,name))
        try:
            for index, address in enumerate([first,second]):
                start(f'stage-{index}', ['stage',str(model),'--first',str(index),'--end',str(index+1),
                    '--model-id','fixture','--context','256','--bind',f'127.0.0.1:{address}'])
            start('gateway',['pipeline-serve','--stages',f'127.0.0.1:{first},127.0.0.1:{second}',
                '--tokenizer',str(model),'--model-id','fixture','--context','256','--bind',f'127.0.0.1:{gateway}'])
            start('local',['serve','--model',str(model),'--bind',f'127.0.0.1:{local}'])
            for address in [gateway,local]:
                for _ in range(100):
                    assert all(p.poll() is None for p,_,_ in processes), 'process exited'
                    try:
                        if call(address,route='/v1/models')[0] == 200: break
                    except OSError: pass
                    time.sleep(0.05)
                else: raise AssertionError('startup timeout')
            for address in [first,second]:
                for _ in range(200):
                    assert all(p.poll() is None for p,_,_ in processes), 'stage exited'
                    try:
                        with socket.create_connection(('127.0.0.1',address), timeout=0.2): pass
                        break
                    except OSError: time.sleep(0.05)
                else: raise AssertionError('stage startup timeout')
            body = {'model':'fixture','messages':[{'role':'user','content':'hello'}],'temperature':0,'max_tokens':4}
            status, reference = call(local,body)
            assert status == 200, reference
            status, actual = call(gateway,body)
            assert status == 200, actual
            reference, actual = json.loads(reference), json.loads(actual)
            assert actual['choices'][0]['message']['content'] == reference['choices'][0]['message']['content'], (actual,reference)
            assert actual['choices'][0]['message']['role'] == reference['choices'][0]['message']['role']
            assert (actual['choices'][0]['message'].get('tool_calls') or []) == (reference['choices'][0]['message'].get('tool_calls') or [])
            assert actual['usage']['completion_tokens'] == reference['usage']['completion_tokens']
            status, stream = call(gateway,dict(body,stream=True))
            assert status == 200 and 'data: [DONE]' in stream, stream
            events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: {')]
            assert ''.join(event['choices'][0]['delta'].get('content','') for event in events) == actual['choices'][0]['message']['content']
            assert len(events) > 2, 'gateway did not stream incremental output'
            assert events[-1]['choices'][0]['finish_reason'] == actual['choices'][0]['finish_reason']
            assert call(gateway,dict(body,model='wrong'))[0] == 400
            assert call(gateway,dict(body,max_tokens=8192))[0] == 400
            assert call(gateway,dict(body,temperature=0.5))[0] == 400
            # Independent requests reconnect and replay without contaminating caches.
            assert json.loads(call(gateway,body)[1])['choices'] == actual['choices']
            processes[1][0].terminate()
            processes[1][0].wait(timeout=5)
            altered = root/'altered.gguf'
            changed = bytearray(model.read_bytes())
            changed[-1] ^= 1
            altered.write_bytes(changed)
            start('mismatched-stage',['stage',str(altered),'--first','1','--end','2','--model-id','fixture',
                '--context','256','--bind',f'127.0.0.1:{second}'])
            for _ in range(200):
                assert processes[-1][0].poll() is None
                try:
                    with socket.create_connection(('127.0.0.1',second), timeout=0.2): pass
                    break
                except OSError: time.sleep(0.05)
            else: raise AssertionError('mismatched stage startup timeout')
            status, rejected = call(gateway,body)
            assert status == 400 and 'different models' in rejected, rejected
            processes[-1][0].terminate()
            processes[-1][0].wait(timeout=5)
            assert call(gateway,body)[0] == 400
            print('PASS: two native stage processes match whole-model chat; JSON/SSE, context, model, replay, changed weights and failed worker checked')
        except Exception:
            for _,log,name in processes:
                log.flush()
                print(name, (root/(name+'.log')).read_text()[-5000:])
            raise
        finally:
            for process,log,_ in processes:
                if process.poll() is None: process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                log.close()

if __name__ == '__main__': main()
