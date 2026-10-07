#!/usr/bin/env python3
import asyncio
import json
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'sdk/python'))
from allpaka_studio import Studio
from allpaka_guardrails import check_guardrails, guard_task, GuardrailBlocked, policy_manifest, rules_from_manifest, save_policy, load_policy

class GuardrailTests(unittest.TestCase):
    def test_required_fragments_all_literal_and_case_sensitive(self):
        rules=[dict(id='required',kind='required_substrings',value=['Привет','ready'])]
        self.assertTrue(check_guardrails('Привет: ready',rules)['passed'])
        self.assertFalse(check_guardrails('Привет',rules)['passed'])
        self.assertFalse(check_guardrails('привет ready',rules)['passed'])
        self.assertEqual(rules_from_manifest(policy_manifest(rules)),rules)

    def test_policy_sdk_routes_and_receipt_integrity(self):
        rules=[dict(id='limit',kind='max_bytes',value=10)]
        manifest=policy_manifest(rules)
        class Client(Studio):
            def __init__(self):self.calls=[];self.receipt=dict(policy=manifest,provider_calls=0)
            def request(self,path,body=None):self.calls.append((path,body));return self.receipt
        client=Client()
        self.assertEqual(client.save_guardrail_policy(rules)['policy'],manifest)
        self.assertEqual(client.calls[0],('guardrail-policies',manifest))
        self.assertEqual(client.create_guardrail_policy(rules)['policy'],manifest)
        self.assertEqual(client.calls[-1],('guardrail-policies/create',dict(rules=rules)))
        client.receipt=dict(provider_calls=0,offset=20,limit=10,total=20,order='policy_sha256_ascending',policies=[])
        client.guardrail_policies(offset=20,limit=10)
        self.assertEqual(client.calls[-1][0],'guardrail-policies?offset=20&limit=10')
        for offset,limit in [(True,20),(0,0),(10001,20),(0,101)]:
            before=len(client.calls)
            with self.assertRaises(ValueError):client.guardrail_policies(offset=offset,limit=limit)
            self.assertEqual(len(client.calls),before)
        client.receipt=dict(policy=manifest,provider_calls=0)
        client.guardrail_policy(manifest['policy_sha256'])
        self.assertEqual(client.calls[-1][0],'guardrail-policies/'+manifest['policy_sha256'])
        for fingerprint in ['../escape','A'*64,None]:
            before=len(client.calls)
            with self.assertRaises(ValueError):client.guardrail_policy(fingerprint)
            self.assertEqual(len(client.calls),before)
        client.receipt=dict(policy=manifest,provider_calls=1)
        with self.assertRaises(ValueError):client.save_guardrail_policy(rules)
        with self.assertRaises(ValueError):client.guardrail_policy(manifest['policy_sha256'])

    def test_task_uses_pinned_server_policies_and_freezes_rules(self):
        rules=[dict(id='limit',kind='max_bytes',value=4)]
        manifest=policy_manifest(rules)
        class Client(Studio):
            def __init__(self):self.calls=0
            def request(self,path,body=None):self.calls+=1;return dict(policy=manifest,provider_calls=0)
        client=Client();executions=[]
        wrapped=client.guard_task_with_policies(lambda text:executions.append(text) or text,input_policy_sha256=manifest['policy_sha256'],output_policy_sha256=manifest['policy_sha256'],action='block')
        self.assertEqual(client.calls,2);manifest['rules'][0]['value']=100
        self.assertEqual(wrapped('safe')['output'],'safe')
        with self.assertRaises(GuardrailBlocked):wrapped('longer')
        self.assertEqual(executions,['safe']);self.assertEqual(client.calls,2)
        with self.assertRaises(ValueError):client.guard_task_with_policies(lambda text:text,input_policy_sha256='bad',output_policy_sha256='bad',action='unknown')
        self.assertEqual(client.calls,2)

    def test_native_check_sdk_explicit_bounds_and_receipts(self):
        rules=[dict(id='limit',kind='max_bytes',value=4)];receipt=check_guardrails('safe',rules,action='observe')
        class Client(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None):self.calls.append((path,body));return receipt
        client=Client();fingerprint=receipt['policy_sha256']
        self.assertEqual(client.check_guardrail_policy(fingerprint,'safe',stage='input',action='observe'),receipt)
        self.assertEqual(client.calls[0][1],dict(text='safe',stage='input',action='observe'))
        for text,stage,action in [('x'*64001,'input','observe'),('safe','unknown','observe'),('safe','input','unknown')]:
            with self.assertRaises(ValueError):client.check_guardrail_policy(fingerprint,text,stage=stage,action=action)
        self.assertEqual(len(client.calls),1)
        receipt['rules'][0]['passed']=False
        with self.assertRaises(ValueError):client.check_guardrail_policy(fingerprint,'safe',stage='input',action='observe')

    def test_trace_rule_receipts_frozen_and_private_fields_rejected(self):
        class Client(Studio):
            def __init__(self):self.packets=[]
            def ingest_trace(self,project_id,correlation_id,started_ms,spans,**kwargs):self.packets.append(dict(spans=spans));return {'id':'saved'}
        client=Client();rules=[dict(id='limit',kind='max_bytes',value=4)]
        with client.trace(project_id='default',correlation_id='receipt-test',name='task') as trace:
            check_guardrails('safe',rules,trace=trace)
        child=client.packets[0]['spans'][1]
        self.assertEqual(child['usage']['guardrail_receipt']['rules'],[dict(rule_id='limit',kind='max_bytes',passed=True)])
        receipt=check_guardrails('safe',rules);name='guardrail.input.observe.pass.'+receipt['policy_sha256']
        with client.trace(project_id='default',correlation_id='receipt-test',name='task') as trace:
            with trace.span(name,'tool') as span:
                span.set_guardrail_receipt(receipt);receipt['rules'][0]['passed']=False
                self.assertTrue(trace.spans[span.index]['usage']['guardrail_receipt']['rules'][0]['passed'])
                changed=check_guardrails('safe',rules);changed['text']='PRIVATE'
                with self.assertRaises(ValueError):span.set_guardrail_receipt(changed)

    def test_policy_file_persistence_exclusive_and_strict(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'policy.json'
            rules=[dict(id='literal',kind='forbidden_substrings',value=['private'])]
            saved=save_policy(path,rules)
            self.assertEqual(load_policy(path),rules)
            from concurrent.futures import ThreadPoolExecutor
            race=path.parent/'race.json'
            def writer(limit):
                try:
                    save_policy(race,[dict(id='limit',kind='max_bytes',value=limit)])
                    return limit
                except FileExistsError:return None
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes=list(pool.map(writer,[10,20]))
            winners=[value for value in outcomes if value is not None]
            self.assertEqual(len(winners),1)
            self.assertEqual(load_policy(race)[0]['value'],winners[0])
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):save_policy(path,rules)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(path.parent.glob('.allpaka-policy-*')),[])
            self.assertEqual(check_guardrails('safe',load_policy(path))['policy_sha256'],saved['policy_sha256'])
            if os.name=='posix':self.assertEqual(path.stat().st_mode & 0o777,0o600)
            path.write_text('{"kind":1,"kind":2}')
            with self.assertRaises(ValueError):load_policy(path)
            path.write_bytes(b'x'*(128*1024+1))
            with self.assertRaises(ValueError):load_policy(path)
            if hasattr(os,'O_NOFOLLOW'):
                linked=path.parent/'link.json';linked.symlink_to(path)
                with self.assertRaises(OSError):load_policy(linked)

    def test_policy_manifest_roundtrip_and_tampering(self):
        rules=[dict(id='literal',kind='forbidden_substrings',value=['secret']),dict(id='limit',kind='max_bytes',value=20)]
        manifest=policy_manifest(rules)
        receipt=check_guardrails('safe',rules)
        self.assertEqual(manifest['policy_sha256'],receipt['policy_sha256'])
        stored=json.loads(json.dumps(manifest))
        restored=rules_from_manifest(stored)
        self.assertEqual(restored,rules)
        restored[0]['value'].append('changed')
        self.assertEqual(stored,manifest)
        stored['rules'][1]['value']=21
        with self.assertRaises(ValueError):rules_from_manifest(stored)
        for field,value in [('schema_version',True),('kind','unknown'),('extra',1)]:
            altered=dict(manifest);altered[field]=value
            with self.assertRaises(ValueError):rules_from_manifest(altered)
        huge=[dict(id='rule'+str(i),kind='forbidden_substrings',value=['x'*1000]*100) for i in range(2)]
        with self.assertRaises(ValueError):policy_manifest(huge)

    def test_strict_json_and_byte_boundaries(self):
        rule=[dict(id='json',kind='json_valid',value=True)]
        for text in ('{"a":1,"a":2}', '{"nested":{"a":1,"a":2}}', 'NaN', 'Infinity', '1e999', '{"a":1e999}', 'invalid', '['*1500):
            self.assertFalse(check_guardrails(text,rule)['passed'])
        self.assertTrue(check_guardrails('{"a":[true,null,1]}',rule)['passed'])
        maximum=[dict(id='limit',kind='max_bytes',value=4)]
        self.assertTrue(check_guardrails('\u044f\u044f',maximum)['passed'])
        self.assertFalse(check_guardrails('\u044f\u044fX',maximum)['passed'])
        with self.assertRaises(ValueError):check_guardrails('X'*64001,maximum)

    def test_block_input_output_observe_and_frozen_policy(self):
        calls=[]
        rules=[dict(id='private',kind='forbidden_substrings',value=['SECRET'])]
        output=[dict(id='json',kind='json_valid',value=True)]
        task=guard_task(lambda text:calls.append(text) or 'PRIVATE OUTPUT',input_rules=rules,output_rules=output)
        rules[0]['value'].clear()
        with self.assertRaises(GuardrailBlocked) as raised:task('SECRET INPUT')
        self.assertEqual(calls,[]);self.assertEqual(raised.exception.receipt['stage'],'input')
        self.assertNotIn('SECRET',json.dumps(raised.exception.receipt));self.assertEqual(str(raised.exception),'guardrail_blocked')
        with self.assertRaises(GuardrailBlocked) as raised:task('allowed')
        self.assertEqual(calls,['allowed']);self.assertEqual(raised.exception.receipt['stage'],'output')
        self.assertTrue(raised.exception.input_receipt['passed'])
        self.assertNotIn('PRIVATE OUTPUT',json.dumps(raised.exception.receipt))
        observed=guard_task(lambda text:'invalid',input_rules=output,output_rules=output,action='observe')('invalid')
        self.assertEqual(observed['output'],'invalid');self.assertTrue(all(not x['passed'] and not x['blocked'] for x in observed['guardrails']))

    def test_business_exception_and_async_cancellation_identity(self):
        rules=[dict(id='length',kind='max_bytes',value=10)]
        original=RuntimeError('PRIVATE ERROR');calls=[]
        def failing(text):calls.append(text);raise original
        with self.assertRaises(RuntimeError) as raised:guard_task(failing,input_rules=rules,output_rules=rules)('input')
        self.assertIs(raised.exception,original);self.assertEqual(calls,['input'])
        async def exercise():
            async def valid(text):return text
            result=await guard_task(valid,input_rules=rules,output_rules=rules)('input')
            self.assertEqual(result['output'],'input')
            cancellation=asyncio.CancelledError()
            async def cancelled(text):raise cancellation
            with self.assertRaises(asyncio.CancelledError) as raised:await guard_task(cancelled,input_rules=rules,output_rules=rules)('input')
            self.assertIs(raised.exception,cancellation)
        asyncio.run(exercise())

    def test_explicit_trace_persists_only_policy_fingerprints_and_outcomes(self):
        class Fake(Studio):
            def __init__(self):self.packets=[]
            def ingest_trace(self,*args,**kwargs):self.packets.append((args,kwargs));return {'id':'saved'}
        client=Fake()
        rules=[dict(id='json',kind='json_valid',value=True)]
        with client.trace('default','guardrail-test') as trace:
            result=guard_task(lambda text:'PRIVATE OUTPUT',input_rules=rules,output_rules=rules,action='observe',trace=trace)('{}')
            self.assertFalse(result['guardrails'][1]['passed'])
            with self.assertRaises(GuardrailBlocked):check_guardrails('PRIVATE INPUT',rules,action='block',trace=trace)
        spans=client.packets[0][0][3]
        self.assertEqual(len(spans),4)
        self.assertEqual([x['status'] for x in spans],['completed','completed','completed','failed'])
        self.assertTrue(all(x['parent_id']==0 for x in spans[1:]))
        self.assertTrue(spans[1]['name'].startswith('guardrail.input.observe.pass.'))
        self.assertTrue(spans[2]['name'].startswith('guardrail.output.observe.fail.'))
        serialized=json.dumps(client.packets)
        self.assertNotIn('PRIVATE',serialized);self.assertIn(result['guardrails'][0]['policy_sha256'],serialized)
        calls=[]
        with self.assertRaises(ValueError):guard_task(lambda text:calls.append(text),input_rules=rules,output_rules=rules,trace=trace)('{}')
        self.assertEqual(calls,[])

    def test_task_evaluation_preserves_blocked_sample_and_partial_outputs(self):
        class Fake(Studio):
            def __init__(self):self.scored=[]
            def _task_preflight(self,*args):return 'data',[dict(id='first',input='{}'),dict(id='second',input='PRIVATE INPUT')],['json_valid'],{},'default'
            def evaluate_outputs(self,*args,**kwargs):self.scored.append(args);return {'passed':True}
        client=Fake();rules=[dict(id='json',kind='json_valid',value=True)];calls=[]
        task=guard_task(lambda text:calls.append(text) or '{}',input_rules=rules,output_rules=rules,return_receipts=False)
        with self.assertRaises(GuardrailBlocked) as raised:client.evaluate_task('data',1,task,['json_valid'])
        self.assertEqual(raised.exception.sample_id,'second')
        self.assertEqual(raised.exception.completed_outputs,{'first':'{}'})
        self.assertEqual(raised.exception.receipt['stage'],'input')
        self.assertEqual(calls,['{}']);self.assertEqual(client.scored,[])
        async def run():
            async def answer(text):return '{}'
            task=guard_task(answer,input_rules=rules,output_rules=rules,return_receipts=False)
            with self.assertRaises(GuardrailBlocked) as raised:await client.evaluate_task_async('data',1,task,['json_valid'],concurrency=1)
            self.assertEqual(raised.exception.sample_id,'second')
            self.assertEqual(raised.exception.completed_outputs,{'first':'{}'})
            self.assertEqual(raised.exception.started_sample_ids,['first','second'])
            self.assertEqual(client.scored,[])
        asyncio.run(run())

    def test_ci_guardrail_evidence_and_junit_exclude_private_fields(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('guardrail_ci',pathlib.Path(__file__).with_name('evaluate-studio.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rules=[dict(id='json',kind='json_valid',value=True)]
        with self.assertRaises(GuardrailBlocked) as raised:check_guardrails('PRIVATE INPUT',rules,action='block')
        error=raised.exception;error.receipt['private']='PRIVATE FIELD'
        evidence=module.guardrail_evidence(error)
        self.assertEqual(evidence['stage'],'input');self.assertNotIn('PRIVATE',json.dumps(evidence))
        suite=module.task_junit({'sample_id':'sample','guardrail':evidence},{'error':'guardrail_blocked'},2)
        xml=module.ET.tostring(suite).decode()
        self.assertIn('guardrail_blocked',xml);self.assertIn(evidence['policy_sha256'],xml);self.assertNotIn('PRIVATE',xml)
        error.receipt['policy_sha256']='PRIVATE HASH'
        self.assertIsNone(module.guardrail_evidence(error))

    def test_cli_configuration_strict_schema_and_duplicate_rejection(self):
        import importlib.util
        import tempfile
        spec=importlib.util.spec_from_file_location('guardrail_config_ci',pathlib.Path(__file__).with_name('evaluate-studio.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=dict(schema_version=1,action='block',input_rules=[dict(id='size',kind='max_bytes',value=100)],output_rules=[dict(id='json',kind='json_valid',value=True)])
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'rules.json';path.write_text(json.dumps(config))
            self.assertEqual(module.load_guardrails(path)['action'],'block')
            input_path=path.parent/'input.json';output_path=path.parent/'output.json'
            save_policy(input_path,config['input_rules']);save_policy(output_path,config['output_rules'])
            options=module.load_policy_guardrails(input_path,output_path,'block')
            self.assertEqual(options,dict(input_rules=config['input_rules'],output_rules=config['output_rules'],action='block'))
            self.assertEqual(guard_task(lambda text:'{}',**options)('safe')['output'],'{}')
            altered=json.loads(output_path.read_text());altered['rules'][0]['value']=False;output_path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):module.load_policy_guardrails(input_path,output_path,'block')
            for raw in ('{"schema_version":1,"schema_version":1}',json.dumps(dict(config,schema_version=True)),json.dumps(dict(config,extra=True)),json.dumps(dict(config,action='unknown')),'X'*(128*1024+1)):
                path.write_text(raw)
                with self.assertRaises(ValueError):module.load_guardrails(path)

    def test_guarded_dataset_span_budget_rejects_before_task_execution(self):
        class Fake(Studio):
            def __init__(self):self.packets=[]
            def _task_preflight(self,*args):return 'data',[dict(id=str(i),input='{}') for i in range(67)],['json_valid'],{},'default'
            def ingest_trace(self,*args,**kwargs):self.packets.append(args);return {'id':'saved'}
        client=Fake();rules=[dict(id='json',kind='json_valid',value=True)];calls=[]
        with client.trace('default','budget') as trace:
            task=guard_task(lambda text:calls.append(text) or '{}',input_rules=rules,output_rules=rules,trace=trace,return_receipts=False)
            with self.assertRaises(ValueError):client.evaluate_task('data',1,task,['json_valid'],trace=trace)
            self.assertEqual(calls,[]);self.assertEqual(len(trace.spans),1)
            client._validate_task_trace(trace,[{}]*66,'default',task)
            nested=guard_task(task,input_rules=rules,output_rules=rules,trace=trace,return_receipts=False)
            with self.assertRaises(ValueError):client._validate_task_trace(trace,[{}]*40,'default',nested)
            async def answer(text):calls.append(text);return '{}'
            wrapped=guard_task(answer,input_rules=rules,output_rules=rules,trace=trace,return_receipts=False)
            async def run():
                with self.assertRaises(ValueError):await client.evaluate_task_async('data',1,wrapped,['json_valid'],trace=trace)
            asyncio.run(run());self.assertEqual(calls,[])

    def test_cli_invalid_configuration_never_loads_task_source(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);marker=root/'executed';task=root/'task.py';config=root/'rules.json';report=root/'report.json'
            task.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("executed")\ndef task(text): return text\n')
            config.write_text('{"schema_version":true}')
            command=[sys.executable,str(pathlib.Path(__file__).with_name('evaluate-studio.py')),'--base-url','http://127.0.0.1:1','--dataset-id','data','--dataset-version','1','--metric','json_valid','--task-file',str(task),'--guardrails-file',str(config),'--report',str(report)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stderr);self.assertFalse(marker.exists())
            self.assertEqual(json.loads(report.read_text())['error'],'invalid_request_or_receipt')

    def test_cli_policy_manifests_preflight_before_task_import(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);marker=root/'executed';task=root/'task.py'
            task.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("executed")\ndef task(text): return text\n')
            input_path=root/'input.json';output_path=root/'output.json'
            rules=[dict(id='length',kind='min_bytes',value=1)]
            save_policy(input_path,rules);save_policy(output_path,rules)
            base=[sys.executable,str(pathlib.Path(__file__).with_name('evaluate-studio.py')),'--base-url','http://127.0.0.1:1','--dataset-id','data','--dataset-version','1','--metric','json_valid','--task-file',str(task)]
            pair=['--input-policy-file',str(input_path),'--output-policy-file',str(output_path),'--policy-action','block']
            for index,changed_path in enumerate([input_path,output_path]):
                original=changed_path.read_bytes();manifest=json.loads(original);manifest['rules'][0]['value']=2;changed_path.write_text(json.dumps(manifest))
                report=root/('report'+str(index)+'.json')
                result=subprocess.run(base+pair+['--report',str(report)],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,2,result.stderr);self.assertFalse(marker.exists())
                self.assertEqual(json.loads(report.read_text())['error'],'invalid_request_or_receipt')
                changed_path.write_bytes(original)
            for flags in [pair[:-2],pair+['--guardrails-file',str(input_path)],['--policy-action','observe']]:
                result=subprocess.run(base+flags,capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,2);self.assertFalse(marker.exists())
                self.assertIn('error:',result.stderr)

    def test_invalid_policies_do_not_execute(self):
        calls=[]
        valid=[dict(id='length',kind='min_bytes',value=1)]
        for rules in ([],[dict(id='a',kind='max_bytes',value=True)], [dict(id='a',kind='unknown',value=True)],valid*2,[dict(id='a',kind='forbidden_substrings',value=[''])]):
            with self.assertRaises(ValueError):guard_task(lambda text:calls.append(text),input_rules=rules,output_rules=valid)
        self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
