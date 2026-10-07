#!/usr/bin/env python3
"""SDK protocol bounds, privacy and fail-closed gates; no model calls."""
import pathlib
import sys
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'sdk/python'))
from allpaka_studio import EvaluationError, Studio


class ClientTests(unittest.TestCase):
    def test_experiment_catalog_protocol_and_bounds(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path):self.calls.append(path);return path
        client=Fake();path=client.experiments('p',status='failed',provider='local',model='MOCK',dataset_id='cases',playground=False,offset=20,limit=10)
        self.assertEqual(path,'evaluation/experiments?project_id=p&offset=20&limit=10&status=failed&provider=local&dataset_id=cases&model=MOCK&playground=false')
        for kwargs in [dict(status='unknown'),dict(offset=True),dict(offset=1001),dict(limit=101),dict(playground=1),dict(model='x'*201),dict(dataset_id='../')]:
            before=len(client.calls)
            with self.assertRaises(ValueError):client.experiments('p',**kwargs)
            self.assertEqual(len(client.calls),before)

    def test_chat_prompt_protocol_freezes_and_bounds(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body):self.calls.append((path,body));return body
        client=Fake();messages=[dict(role='user',content='Example'),dict(role='assistant',content='Answer'),dict(role='user',content='{{input}}')]
        receipt=client.save_chat_prompt('p','Chat',messages);messages[0]['content']='Changed';self.assertEqual(receipt['messages'][0]['content'],'Example');self.assertEqual(receipt['template'],'')
        for invalid in [[],[dict(role='assistant',content='{{input}}')],[dict(role='user',content='Question')],messages[:2],[dict(role='user',content='{{input}}',extra=True)],[dict(role='user',content='x'*16001+'{{input}}')]]:
            before=len(client.calls)
            with self.assertRaises(ValueError):client.save_chat_prompt('p','Chat',invalid)
            self.assertEqual(len(client.calls),before)

    def test_playground_protocol_freezes_and_validates(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body):self.calls.append((path,body));return body
        client=Fake();settings=dict(project_id='p',provider='local',model='mock',mode='chat',allow_writes=False);contexts=['Context']
        result=client.start_playground('prompt',2,'a'*64,'Question',settings,contexts=contexts);settings['model']='changed';contexts.clear()
        self.assertEqual(client.calls[0][0],'evaluation/playground');self.assertEqual(result['settings']['model'],'mock');self.assertEqual(result['contexts'],['Context']);self.assertEqual(result['metrics'],[])
        chosen=['exact_match','whitespace_token_f1'];scored=client.start_playground('prompt',2,'a'*64,'Question',settings,expected_output='Reference',metrics=chosen);chosen.clear()
        self.assertEqual(scored['metrics'],['exact_match','whitespace_token_f1']);self.assertEqual(scored['expected_output'],'Reference')

        for changes in [dict(prompt_version=True),dict(prompt_sha256='bad'),dict(input_text=''),dict(contexts=[{}]),dict(metrics=['exact_match']),dict(metrics=['json_valid','json_valid']),dict(item_timeout_secs=0),dict(settings=dict(settings,allow_writes=True))]:
            arguments=dict(prompt_id='prompt',prompt_version=2,prompt_sha256='a'*64,input_text='Question',settings=settings);arguments.update(changes);before=len(client.calls)
            with self.assertRaises(ValueError):client.start_playground(**arguments)
            self.assertEqual(len(client.calls),before)

    def test_prompt_preview_protocol(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body):self.calls.append((path,body));return body
        client=Fake();contexts=['One','Two'];result=client.preview_prompt('prompt',2,'project','Question',contexts=contexts);contexts.clear()
        self.assertEqual(client.calls[0][0],'evaluation/prompts/prompt/versions/2/preview');self.assertEqual(result['contexts'],['One','Two'])
        for version,text,ctx in [(True,'OK',[]),(0,'OK',[]),(1,'',[]),(1,'я'*32769,[]),(1,'OK',['x']*51),(1,'OK',[{}])]:
            before=len(client.calls)
            with self.assertRaises(ValueError):client.preview_prompt('prompt',version,'project',text,contexts=ctx)
            self.assertEqual(len(client.calls),before)

    def test_dataset_comparison_csv_pages_and_pins(self):
        import copy,csv,io
        changes=[dict(sample_id='s'+str(i).zfill(3),kind='added',fields=[]) for i in range(101)]
        class Fake(Studio):
            def __init__(self):self.calls=[];self.mutate=None
            def compare_dataset_versions(self,dataset_id,project_id,a,b,*,offset=0,limit=100):
                self.calls.append(offset)
                page=dict(id=dataset_id,project_id=project_id,provider_calls=0,offset=offset,limit=limit,order='sample_id_asc',total=101,has_more=offset==0,counts=dict(added=101,removed=0,changed=0,unchanged=0),changes=copy.deepcopy(changes[offset:offset+limit]),**{'from':dict(version=a,sha256='a'*64),'to':dict(version=b,sha256='b'*64)})
                if self.mutate:self.mutate(page)
                return page
        client=Fake();rows=list(csv.DictReader(io.StringIO(client.export_dataset_comparison_csv('d','p',1,2))))
        self.assertEqual(len(rows),101);self.assertEqual(client.calls,[0,100]);self.assertEqual(rows[-1]['sample_id'],'s100');self.assertEqual(rows[-1]['from_sha256'],'a'*64)
        for mutate in [lambda p:p['to'].update(sha256='c'*64) if p['offset'] else None,lambda p:p.update(total=5000),lambda p:p['changes'][0].update(fields=[{}]),lambda p:p['changes'][0].update(sample_id='s000') if p['offset'] else None,lambda p:p['counts'].update(changed=1),lambda p:p.update(project_id='foreign')]:
            client=Fake();client.mutate=mutate
            with self.assertRaises(EvaluationError):client.export_dataset_comparison_csv('d','p',1,2)
        client=Fake();client.mutate=lambda p:p['changes'][0].update(sample_id='-formula') if not p['offset'] else None
        rows=list(csv.DictReader(io.StringIO(client.export_dataset_comparison_csv('d','p',1,2))))
        self.assertEqual(rows[0]['sample_id'],"'-formula")

    def test_dataset_comparison_protocol(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path):self.calls.append(path);return path
        client=Fake();self.assertEqual(client.compare_dataset_versions('d','p',1,2,offset=100,limit=20),'evaluation/datasets/d/compare?project_id=p&from_version=1&to_version=2&offset=100&limit=20')
        for a,b,offset,limit in [(True,2,0,20),(0,2,0,20),(1,1001,0,20),(1,2,-1,20),(1,2,4001,20),(1,2,0,101),(1,2,0,False)]:
            before=len(client.calls)
            with self.assertRaises(ValueError):client.compare_dataset_versions('d','p',a,b,offset=offset,limit=limit)
            self.assertEqual(len(client.calls),before)

    def test_dataset_variant_protocol(self):
        class Fake(Studio):
            def __init__(self):self.calls=[];self.source=dict(id='d',version=1,project_id='p',sha256='a'*64,samples=[dict(id='s',input='original')])
            def request(self,path,body=None):self.calls.append((path,body));return self.source if body is None else body
        client=Fake();result=client.fork_dataset('d',1,'p','Variant')
        self.assertEqual(result['origin'],dict(id='d',version=1,sha256='a'*64));self.assertNotIn('id',result)
        self.assertEqual(client.calls[0][0],'evaluation/datasets/d/versions/1')
        for version,name in [(True,'OK'),(0,'OK'),(1001,'OK'),(1,''),(1,'я'*101)]:
            before=len(client.calls)
            with self.assertRaises(ValueError):client.fork_dataset('d',version,'p',name)
            self.assertEqual(len(client.calls),before)
        client.source['project_id']='foreign'
        with self.assertRaises(ValueError):client.fork_dataset('d',1,'p','Variant')
        self.assertIsNone(client.calls[-1][1])

    def test_dataset_versions_protocol(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path):self.calls.append(path);return path
        client=Fake();self.assertEqual(client.dataset_versions('d','p',offset=20,limit=10),'evaluation/datasets/d/versions?project_id=p&offset=20&limit=10')
        for offset,limit in [(True,20),(-1,20),(1001,20),(0,0),(0,101)]:
            with self.assertRaises(ValueError):client.dataset_versions('d','p',offset=offset,limit=limit)
        self.assertEqual(len(client.calls),1)

    def test_dataset_archive_protocol(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None):self.calls.append((path,body));return path if body is None else body
        client=Fake();self.assertEqual(client.datasets('p',archived=True),'evaluation/datasets?project_id=p&archived=true')
        self.assertEqual(client.dataset_lifecycle('d','p',base_version=1,base_revision=0,archived=True)['archived'],True)
        self.assertEqual(client.datasets('p',offset=20,limit=20),'evaluation/datasets?project_id=p&archived=false&offset=20&limit=20')
        self.assertIn('q=Fixture',client.datasets('p',search='Fixture'))
        before=len(client.calls)
        for search in [True,'x'*201]:
            with self.assertRaises(ValueError):client.datasets('p',search=search)
        self.assertEqual(len(client.calls),before)
        before=len(client.calls)
        for offset,limit in [(True,20),(-1,20),(2001,20),(0,0),(0,101)]:
            with self.assertRaises(ValueError):client.datasets('p',offset=offset,limit=limit)
        self.assertEqual(len(client.calls),before)
        before=len(client.calls)
        for version,revision,archived in [(True,0,True),(0,0,True),(1,-1,True),(1,True,True),(1,0,1)]:
            with self.assertRaises(ValueError):client.dataset_lifecycle('d','p',base_version=version,base_revision=revision,archived=archived)
        self.assertEqual(len(client.calls),before)

    def test_conversation_summary_pagination_bounds(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return path
        client=Fake()
        self.assertEqual(client.trace_summary(project_id='project',conversation_offset=100,conversation_limit=50),'observability/summary?project_id=project&conversation_offset=100&conversation_limit=50')
        self.assertEqual(client.trace_summary(status='goal_incomplete'),'observability/summary?status=goal_incomplete')
        before=len(client.calls)
        for status in ['',True,'FAILED','bad-status','a'*41]:
            with self.assertRaises(ValueError):client.trace_summary(status=status)
        self.assertEqual(len(client.calls),before)
        before=len(client.calls)
        for offset,limit in [(True,10),(-1,10),(10001,10),(0,False),(0,0),(0,101)]:
            with self.assertRaises(ValueError):client.trace_summary(conversation_offset=offset,conversation_limit=limit)
        self.assertEqual(len(client.calls),before)

    def test_durable_session_plan_bounds_revision_and_freezing(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return body if body is not None else path
        client=Fake();self.assertEqual(client.session_plan('session'),'sessions/session/plan')
        steps=[dict(title='Этап',status='pending',acceptance=['Критерий'],evidence=[])]
        packet=client.update_session_plan('session',steps,base_revision=2);steps[0]['title']='changed';self.assertEqual(packet['steps'][0]['title'],'Этап');self.assertFalse(packet['allow_reopen'])
        self.assertTrue(client.update_session_plan('session',steps,base_revision=2,allow_reopen=True)['allow_reopen'])
        before=len(client.calls)
        for revision,reopen in [(True,False),(-1,False),(2**64,False),(0,1)]:
            with self.assertRaises(ValueError):client.update_session_plan('session',steps,base_revision=revision,allow_reopen=reopen)
        for item in [dict(title='x',status='done'),dict(title='x',status='pending',id=None),dict(title='x',status='pending',acceptance=['a']*5),dict(title='я'*251,status='pending'),dict(title='x',status='pending',evidence=['a'*2001])]:
            with self.assertRaises(ValueError):client.update_session_plan('session',[item],base_revision=0)
        self.assertEqual(len(client.calls),before)

    def test_server_matrix_job_protocol_and_bounds(self):
        import copy
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return body if body is not None else path
        client=Fake();variants=[dict(label=label,sha256='a'*64,request=dict(settings=dict(project_id='project'))) for label in ['base','next']]
        frozen=client.start_matrix_job('project',variants);variants[0]['label']='changed';self.assertEqual(frozen['variants'][0]['label'],'base')
        self.assertEqual(client.matrix_job('job'),'evaluation/matrix-jobs/job')
        self.assertEqual(client.matrix_jobs('project',offset=1,limit=2),'evaluation/matrix-jobs?project_id=project&offset=1&limit=2')
        client.cancel_matrix_job('job');client.resume_matrix_job('job');self.assertEqual(client.calls[-1],('evaluation/matrix-jobs/job/resume',{}))
        client.retry_matrix_job('job');self.assertEqual(client.calls[-1],('evaluation/matrix-jobs/job/retry',{}))
        for offset,limit in [(True,1),(1001,1),(-1,1),(0,0),(0,True)]:
            with self.assertRaises(ValueError):client.matrix_jobs('project',offset=offset,limit=limit)
        valid=copy.deepcopy(frozen['variants']);pinned=copy.deepcopy(valid);pinned[1]['prompt_sha256']='b'*64;self.assertEqual(client.start_matrix_job('project',pinned)['variants'][1]['prompt_sha256'],'b'*64);before=len(client.calls)
        for rows in [valid[:1],[valid[0],valid[0]],[valid[0],dict(valid[1],sha256='invalid')],[valid[0],dict(valid[1],prompt_sha256='invalid')],[valid[0],dict(valid[1],request=dict(settings=dict(project_id='other')))]]:
            with self.assertRaises(ValueError):client.start_matrix_job('project',rows)
        self.assertEqual(len(client.calls),before)

    def test_matrix_catalog_page_bounds(self):
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return path
        client=Fake();self.assertEqual(client.experiment_matrices('project',offset=2,limit=1),'evaluation/matrices?project_id=project&offset=2&limit=1')
        for offset,limit in [(True,1),(2001,1),(-1,1),(0,101),(0,0)]:
            with self.assertRaises(ValueError):client.experiment_matrices('project',offset=offset,limit=limit)

    def test_native_matrix_save_bounds_and_read(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return body or path
        client=Fake();variants=[dict(label='База',run_id='base'),dict(label='Candidate',run_id='next')]
        result=client.save_experiment_matrix('project','base',variants)
        self.assertEqual(result['variants'],variants);self.assertEqual(client.experiment_matrix('matrix'),'evaluation/matrices/matrix')
        for rows in [variants[:1],variants[::-1],[variants[0],variants[0]],[variants[0],dict(label='я'*101,run_id='next')]]:
            with self.assertRaises(ValueError):client.save_experiment_matrix('project','base',rows)
        self.assertEqual(len(client.calls),2)

    def test_comparison_csv_gate_and_formula_validation(self):
        import csv,io,copy
        packet=dict(id='comparison',baseline_id='base',candidate_id='next',project_id='project',dataset_id='dataset',dataset_sha256='hash',dataset_version=1,observational_only=True,eligible=False,regressions=1,improvements=1,reason='paired_regression',pairs=[dict(sample_id='=1+1',metric='exact_match',baseline=1,candidate=0,delta=-1),dict(sample_id='Привет',metric='exact_match',baseline=0,candidate=1,delta=1)])
        class Fake(Studio):
            def comparison(self,comparison_id):return packet
        rows=list(csv.DictReader(io.StringIO(Fake().export_comparison_csv('comparison'))))
        self.assertEqual(rows[0]['sample_id'],"'=1+1");self.assertEqual(rows[0]['delta'],'-1')
        self.assertEqual(rows[0]['change'],'regression');self.assertEqual(rows[1]['change'],'improvement')
        self.assertEqual(rows[1]['sample_id'],'Привет');self.assertEqual(rows[1]['eligible'],'False')
        original=copy.deepcopy(packet)
        for key,value in [('eligible',True),('regressions',0),('reason','strict_paired_improvement')]:
            packet[key]=value
            with self.assertRaises(EvaluationError):Fake().export_comparison_csv('comparison')
            packet.clear();packet.update(copy.deepcopy(original))
        packet['pairs'][0]['delta']=0
        with self.assertRaises(EvaluationError):Fake().export_comparison_csv('comparison')

    def test_comparison_catalog_bounds(self):
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return path
        client=Fake();self.assertEqual(client.comparisons("default",offset=2,limit=1),"evaluation/comparisons?project_id=default&offset=2&limit=1")
        for offset,limit in [(True,1),(-1,1),(2001,1),(0,0),(0,101),(0,True)]:
            with self.assertRaises(ValueError):client.comparisons("default",offset=offset,limit=limit)

    def test_saved_comparison_route_and_id_bounds(self):
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return path
        client=Fake();self.assertEqual(client.comparison('saved'),'evaluation/comparisons/saved')
        with self.assertRaises(ValueError):client.comparison('../escape')

    def test_experiment_csv_preserves_failures_unicode_and_neutralizes_formulas(self):
        import csv,io,copy
        packet=dict(kind='experiment_export',schema_version=1,run_id='run',outputs_included=False,
            dataset_id='dataset',dataset_version=2,dataset_sha256='hash',status='failed',strict_quality=False,
            metrics=['exact_match'],items=[dict(sample_id='one',status='completed',duration_ms=7,has_error=False,
                output_truncated=False,scores={'exact_match':1},output='Привет,\n"мир"'),
                dict(sample_id='=1+1',status='failed',duration_ms=None,has_error=True,output_truncated=True,
                scores={},output='  @SUM(A1)')])
        class Fake(Studio):
            def export_experiment(self,run_id,*,include_outputs=False):
                result=copy.deepcopy(packet);result['outputs_included']=include_outputs;return result
        client=Fake()
        packet.update(provider='local',model='=MODEL',trace_id='trace',prompt_ref=dict(id='prompt',version=3,sha256='a'*64))
        rows=list(csv.DictReader(io.StringIO(client.export_experiment_csv('run'))))
        self.assertEqual(rows[0]['model'],"'=MODEL");self.assertEqual(rows[0]['prompt_version'],'3')
        self.assertEqual(rows[0]['prompt_sha256'],'a'*64);self.assertEqual(rows[0]['trace_id'],'trace')
        packet['prompt_ref']=dict(id='prompt',version=True,sha256='hash')
        with self.assertRaises(EvaluationError):client.export_experiment_csv('run')
        packet['prompt_ref']=None
        rows=list(csv.DictReader(io.StringIO(client.export_experiment_csv('run'))))
        self.assertEqual(rows[0]['prompt_id'],'')
        self.assertEqual(len(rows),2);self.assertNotIn('output',rows[0])
        self.assertEqual(rows[1]['sample_id'],"'=1+1");self.assertEqual(rows[1]['score_exact_match'],'')
        self.assertEqual(rows[1]['duration_ms'],'');self.assertEqual(rows[1]['run_status'],'failed')
        rows=list(csv.DictReader(io.StringIO(client.export_experiment_csv('run',include_outputs=True))))
        self.assertEqual(rows[0]['output'],'Привет,\n"мир"');self.assertEqual(rows[1]['output'],"'  @SUM(A1)")
        for value in [float('nan'),True,2]:
            packet['items'][0]['scores']['exact_match']=value
            with self.assertRaises(EvaluationError):client.export_experiment_csv('run')

    def test_review_queue_completion_requests(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.complete_review_queue('queue-one',0,3,'annotation-one',base_version=2)
        self.assertEqual(client.calls[0][1],dict(base_version=2,target_index=0,action='complete',feedback_version=3,annotation_id='annotation-one'))
        client.reopen_review_queue('queue-one',0,base_version=3)
        self.assertEqual(client.calls[1][1],dict(base_version=3,target_index=0,action='reopen'))
        for version in [0,True,2001]:
            with self.assertRaises(ValueError):client.complete_review_queue('queue-one',0,version,'annotation-one',base_version=2)
        for index,version in [(True,2),(200,2),(0,False),(0,0)]:
            with self.assertRaises(ValueError):client.reopen_review_queue('queue-one',index,base_version=version)
        self.assertEqual(len(client.calls),2)

    def test_review_queue_assignment_validation(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.assign_review_queue('queue-one',0,'Reviewer',base_version=1);client.assign_review_queue('queue-one',0,None,base_version=2)
        self.assertIsNone(client.calls[1][1]['reviewer'])
        for index,name,version in [(True,'Reviewer',1),(200,'Reviewer',1),(0,'',1),(0,'я'*101,1),(0,'Reviewer',True),(0,'Reviewer',0)]:
            with self.assertRaises(ValueError):client.assign_review_queue('queue-one',index,name,base_version=version)
        self.assertEqual(len(client.calls),2)

    def test_model_evaluator_configuration_preflight(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.save_online_model_evaluator({},'criterion');client.online_model_evaluator('default','a'*64)
        self.assertEqual(client.calls[0],('observability/online-model-evaluators',dict(settings={},rubric='criterion')))
        self.assertEqual(client.calls[1][0],'observability/online-model-evaluators/'+'a'*64+'?project_id=default')
        for rubric in ['', ' ', 'я'*8001]:
            with self.assertRaises(ValueError):client.save_online_model_evaluator({},rubric)
        self.assertEqual(len(client.calls),2)

    def test_quality_job_catalog_preflight(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.online_quality_jobs('default',offset=1,limit=1)
        self.assertEqual(client.calls[0][0],'observability/online-quality-jobs?project_id=default&offset=1&limit=1')
        for offset,limit in [(True,20),(-1,20),(1001,20),(0,True),(0,0),(0,101)]:
            with self.assertRaises(ValueError):client.online_quality_jobs('default',offset=offset,limit=limit)
        self.assertEqual(len(client.calls),1)

    def test_quality_job_submission_and_read_preflight(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.submit_online_quality_job('a'*64,{},'rubric');client.online_quality_job('b'*64)
        self.assertEqual(client.calls[0][0],'observability/online-quality-jobs');self.assertEqual(client.calls[1][0],'observability/online-quality-jobs/'+'b'*64)
        for rubric in ['', ' ', 'я'*8001]:
            with self.assertRaises(ValueError):client.submit_online_quality_job('a'*64,{},rubric)
        for job_id in ['', 'A'*64]:
            with self.assertRaises(ValueError):client.online_quality_job(job_id)
        self.assertEqual(len(client.calls),2)

    def test_quality_source_storage_preflight(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.save_online_quality_source('default','trace-one','a'*64,'question','answer');client.online_quality_source('default','b'*64);client.judge_online_quality_source('b'*64,{},'rubric')
        self.assertEqual(client.calls[0][1]['output'],'answer');self.assertEqual(client.calls[1][0],'observability/online-quality-sources/'+'b'*64+'?project_id=default')
        for output in ['', ' ', 'я'*32001]:
            with self.assertRaises(ValueError):client.save_online_quality_source('default','trace-one','a'*64,'question',output)
        self.assertEqual(client.calls[2][0],'observability/online-quality-sources/'+'b'*64+'/judge');self.assertEqual(client.calls[2][1],dict(settings={},rubric='rubric'))
        self.assertEqual(len(client.calls),3)

    def test_trace_judge_pins_explicit_text_and_source(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body,timeout));return {}
        client=Fake();client.judge_trace('trace-one','a'*64,{'project_id':'default'},'rubric','question','answer')
        path,body,timeout=client.calls[0];self.assertEqual(path,'evaluation/judge');self.assertEqual(timeout,65);self.assertEqual(body['trace_ref'],dict(trace_id='trace-one',trace_sha256='a'*64));self.assertEqual(body['output'],'answer')
        for fingerprint in ['', 'A'*64,'a'*63]:
            with self.assertRaises(ValueError):client.judge_trace('trace-one',fingerprint,{},'rubric','question','answer')
        self.assertEqual(len(client.calls),1)

    def test_online_job_catalog_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.online_evaluation_jobs('default',offset=1,limit=1)
        self.assertEqual(client.calls,[('observability/online-jobs?project_id=default&offset=1&limit=1',None)])
        for offset,limit in [(True,20),(-1,20),(1001,20),(0,True),(0,0),(0,101)]:
            with self.assertRaises(ValueError):client.online_evaluation_jobs('default',offset=offset,limit=limit)
        self.assertEqual(len(client.calls),1)

    def test_online_job_batch_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.drain_online_evaluation_jobs(limit=1)
        self.assertEqual(client.calls,[('observability/online-jobs/drain',dict(limit=1))])
        for limit in [True,0,101]:
            with self.assertRaises(ValueError):client.drain_online_evaluation_jobs(limit=limit)
        self.assertEqual(len(client.calls),1)

    def test_online_assessment_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.assess_online_trace('default','trace-one')
        self.assertEqual(client.calls,[('observability/online-assessments',dict(project_id='default',trace_id='trace-one'))])
        with self.assertRaises(ValueError):client.assess_online_trace('default','../escape')
        self.assertEqual(len(client.calls),1)

    def test_online_selection_archive_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.online_evaluation_selection_archive('default','trace-one')
        self.assertEqual(client.calls,[('observability/online-selections?project_id=default&trace_id=trace-one',None)])
        with self.assertRaises(ValueError):client.online_evaluation_selection_archive('default','../escape')
        self.assertEqual(len(client.calls),1)

    def test_online_selection_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.select_online_evaluations('default','trace-one')
        self.assertEqual(client.calls,[('observability/online-selections',dict(project_id='default',trace_id='trace-one'))])
        with self.assertRaises(ValueError):client.select_online_evaluations('default','../escape')
        self.assertEqual(len(client.calls),1)

    def test_online_binding_requests(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.bind_online_evaluation_rule('a'*64,base_version=0,active=True);client.online_evaluation_binding('default','quality')
        self.assertEqual(client.calls[0],('observability/online-rule-bindings',dict(rule_sha256='a'*64,base_version=0,active=True)))
        self.assertEqual(client.calls[1],('observability/online-rule-bindings?project_id=default&rule_id=quality',None))
        for version,active in [(True,True),(1000,True),(0,1)]:
            with self.assertRaises(ValueError):client.bind_online_evaluation_rule('a'*64,base_version=version,active=active)
        self.assertEqual(len(client.calls),2)

    def test_online_rule_catalog_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.online_evaluation_rules('default',offset=1,limit=1)
        self.assertEqual(client.calls,['observability/online-rules?project_id=default&offset=1&limit=1'])
        for offset,limit in [(True,1),(1001,1),(0,0),(0,101)]:
            with self.assertRaises(ValueError):client.online_evaluation_rules('default',offset=offset,limit=limit)
        self.assertEqual(len(client.calls),1)

    def test_online_rule_storage_requests(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();rule=dict(id='quality',project_id='default',evaluator_id='rubric',evaluator_version=1,sample_rate=1,enabled=True)
        client.save_online_evaluation_rule(rule);client.online_evaluation_rule('a'*64)
        self.assertEqual(client.calls[0],('observability/online-rules',dict(rule,sample_rate=1.0)))
        self.assertEqual(client.calls[1],('observability/online-rules/'+'a'*64,None))
        for change in [dict(sample_rate=float('nan')),dict(sample_rate=True),dict(evaluator_version=0),dict(enabled=1),dict(id='../escape'),dict(prompt='extra')]:
            with self.assertRaises(ValueError):client.save_online_evaluation_rule(dict(rule,**change))
        with self.assertRaises(ValueError):client.online_evaluation_rule('../escape')
        self.assertEqual(len(client.calls),2)

    def test_memory_extraction_source_status_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return dict(status='current')
        client=Fake()
        self.assertEqual(client.memory_extraction_source_status('proposal-one'),dict(status='current'))
        self.assertEqual(client.calls,[('memory/proposals/proposal-one/source-status',None)])
        for identity in ['', '../escape']:
            with self.assertRaises(ValueError):client.memory_extraction_source_status(identity)
        self.assertEqual(len(client.calls),1)

    def test_memory_consolidation_catalog_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.memory_consolidation_proposals('default',offset=1,limit=1);self.assertEqual(client.calls,['memory/consolidation-proposals?project_id=default&offset=1&limit=1'])
        for offset,limit in [(True,1),(1001,1),(0,0),(0,101)]:
            with self.assertRaises(ValueError):client.memory_consolidation_proposals('default',offset=offset,limit=limit)
        self.assertEqual(len(client.calls),1)

    def test_model_memory_consolidation_requests(self):
        pins=[dict(id='first',version=1,sha256='a'*64),dict(id='second',version=1,sha256='b'*64)]
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):
                self.calls.append((path,body,timeout))
                if path.startswith('memory/proposals/'):return dict(kind='memory_consolidation_proposal',id='proposal-one',project_id='default',notes=[dict(name='Combined',content='Reviewed')],consolidation_sources=pins)
                return body
        client=Fake();settings=dict(project_id='default',mode='chat',allow_writes=False)
        client.propose_memory_consolidation(settings,pins);self.assertEqual(client.calls[0],('memory/consolidation-proposals',dict(settings=settings,sources=pins),65))
        accepted=client.accept_memory('proposal-one',0);self.assertEqual(accepted['consolidation_sources'],pins);self.assertEqual(accepted['proposal_source'],dict(proposal_id='proposal-one',note_index=0))
        calls=len(client.calls)
        for invalid in [pins[:1],[pins[0],pins[0]],[dict(pins[0],version=True),pins[1]]]:
            with self.assertRaises(ValueError):client.propose_memory_consolidation(settings,invalid)
        with self.assertRaises(ValueError):client.propose_memory_consolidation(dict(settings,allow_writes=True),pins)
        self.assertEqual(len(client.calls),calls)

    def test_memory_source_status_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.memory_consolidation_source_status('note-one',2);self.assertEqual(client.calls,['memory/notes/note-one/versions/2/source-status'])
        for version in (True,0,1001):
            with self.assertRaises(ValueError):client.memory_consolidation_source_status('note-one',version)
        self.assertEqual(len(client.calls),1)

    def test_memory_consolidation_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return body
        client=Fake();sources=[dict(id='first',version=1,sha256='a'*64),dict(id='second',version=2,sha256='b'*64)]
        receipt=client.consolidate_memory('default',sources,name='Combined',content='Reviewed content');self.assertEqual(receipt['consolidation_sources'],sources);self.assertEqual(receipt['base_version'],0)
        for invalid in [sources[:1],[sources[0],sources[0]],[dict(sources[0],version=True),sources[1]],[dict(sources[0],sha256='bad'),sources[1]]]:
            with self.assertRaises(ValueError):client.consolidate_memory('default',invalid,name='Combined',content='Reviewed content')
        self.assertEqual(len(client.calls),1)

    def test_callback_summary_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.callback_evaluation_summary('default',since_ms=1,until_ms=2)
        self.assertEqual(client.calls,['observability/evaluations/summary?project_id=default&since_ms=1&until_ms=2'])
        for bounds in [dict(since_ms=True),dict(until_ms=-1),dict(since_ms=2,until_ms=1)]:
            with self.assertRaises(ValueError):client.callback_evaluation_summary('default',**bounds)
        self.assertEqual(len(client.calls),1)

    def test_callback_summary_exports(self):
        import copy,csv,io,json
        packet=dict(kind='callback_evaluation_summary',project_id='default',since_ms=1,until_ms=2,assessment_source='caller_reported',provider_calls=0,automatic_promotion=False,trace_count=3,selected_tasks=2,skipped_tasks=1,completed_assessments=1,failed_assessments=1,metrics=[dict(evaluator_id='-check',evaluator_version=2,metric='metric'+str(i),count=1,mean=.5,min=.5,max=.5) for i in range(51)])
        class Fake(Studio):
            def __init__(self):self.calls=[];self.packet=copy.deepcopy(packet)
            def request(self,path,body=None,timeout=10):self.calls.append(path);return self.packet
        client=Fake();rows=list(csv.DictReader(io.StringIO(client.export_callback_evaluation_summary('default',format='csv',since_ms=1,until_ms=2))))
        self.assertEqual(len(rows),51);self.assertEqual(rows[-1]['metric'],'metric50');self.assertEqual(rows[0]['evaluator_id'],"'-check");self.assertEqual(rows[0]['selected_tasks'],'2');self.assertEqual(len(client.calls),1)
        self.assertEqual(json.loads(client.export_callback_evaluation_summary('default',since_ms=1,until_ms=2)),packet)
        client.packet['metrics']=[];rows=list(csv.DictReader(io.StringIO(client.export_callback_evaluation_summary('default',format='csv',since_ms=1,until_ms=2))));self.assertEqual(len(rows),1);self.assertEqual(rows[0]['metric'],'');self.assertEqual(rows[0]['failed_assessments'],'1')
        for field,value in [('project_id','foreign'),('since_ms',0),('since_ms',True),('provider_calls',False),('automatic_promotion',True),('trace_count',True)]:
            client.packet=copy.deepcopy(packet);client.packet[field]=value
            with self.assertRaises(EvaluationError):client.export_callback_evaluation_summary('default',since_ms=1,until_ms=2)
        for field,value in [('mean',True),('mean',float('nan')),('mean',10**1000),('mean',.6),('count',0),('evaluator_version',True),('metric','private text')]:
            client.packet=copy.deepcopy(packet);client.packet['metrics'][0][field]=value
            with self.assertRaises(EvaluationError):client.export_callback_evaluation_summary('default',since_ms=1,until_ms=2)
        client.packet=copy.deepcopy(packet);client.packet['metrics'].append(client.packet['metrics'][0])
        with self.assertRaises(EvaluationError):client.export_callback_evaluation_summary('default',since_ms=1,until_ms=2)
        calls=len(client.calls)
        with self.assertRaises(ValueError):client.export_callback_evaluation_summary('default',format='xml')
        self.assertEqual(len(client.calls),calls)

    def test_callback_summary_evaluator_attempts(self):
        import copy,json
        packet=dict(kind='callback_evaluation_summary',project_id='default',since_ms=None,until_ms=None,assessment_source='caller_reported',provider_calls=0,automatic_promotion=False,trace_count=3,selected_tasks=2,skipped_tasks=1,completed_assessments=1,failed_assessments=2,metrics=[],evaluators=[dict(evaluator_id='check',evaluator_version=2,completed_assessments=1,failed_assessments=1),dict(evaluator_id=None,evaluator_version=None,completed_assessments=0,failed_assessments=1)])
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return self.packet
        client=Fake('http://127.0.0.1:1');client.packet=copy.deepcopy(packet)
        self.assertEqual(json.loads(client.export_callback_evaluation_summary('default')),packet)
        import csv,io
        rows=list(csv.DictReader(io.StringIO(client.export_callback_evaluator_attempts_csv('default'))))
        self.assertEqual(len(rows),2);self.assertEqual(rows[1]['failed_assessments'],'1');self.assertEqual(rows[1]['evaluator_id'],'');self.assertEqual(rows[0]['evaluator_version'],'2')
        client.packet=dict(packet,evaluators=[],completed_assessments=0,failed_assessments=0)
        rows=list(csv.DictReader(io.StringIO(client.export_callback_evaluator_attempts_csv('default'))));self.assertEqual(len(rows),1);self.assertEqual(rows[0]['project_id'],'default');self.assertEqual(rows[0]['failed_assessments'],'')
        client.packet=dict(packet);del client.packet['evaluators']
        with self.assertRaises(EvaluationError):client.export_callback_evaluator_attempts_csv('default')

        for field,value in [('failed_assessments',True),('failed_assessments',0),('evaluator_version',None)]:
            client.packet=copy.deepcopy(packet);client.packet['evaluators'][0][field]=value
            with self.assertRaises(EvaluationError):client.export_callback_evaluation_summary('default')
        client.packet=copy.deepcopy(packet);client.packet['evaluators'].append(client.packet['evaluators'][0])
        with self.assertRaises(EvaluationError):client.export_callback_evaluation_summary('default')

    def test_callback_summary_gate(self):
        import copy
        packet=dict(kind='callback_evaluation_summary',project_id='default',since_ms=None,until_ms=None,assessment_source='caller_reported',provider_calls=0,automatic_promotion=False,trace_count=3,selected_tasks=3,skipped_tasks=0,completed_assessments=2,failed_assessments=1,metrics=[dict(evaluator_id='check',evaluator_version=2,metric='quality',count=2,mean=.75,min=.5,max=1)])
        class Fake(Studio):
            def __init__(self):self.calls=0
            def request(self,path,body=None,timeout=10):self.calls+=1;return packet
        client=Fake();requirement=dict(evaluator_id='check',evaluator_version=2,metric='quality',min_count=2,min_mean=.75,min_score=.5)
        result=client.check_callback_evaluation_summary('default',[requirement],max_failed_assessments=1)
        self.assertTrue(result['passed']);self.assertEqual(client.calls,1);self.assertEqual(result['summary'],packet);self.assertFalse(result['automatic_promotion'])
        self.assertFalse(client.check_callback_evaluation_summary('default',[requirement],max_failed_assessments=0)['passed'])
        for key,value,reason in [('evaluator_version',1,'missing_metric_group'),('min_count',3,'insufficient_samples'),('min_mean',.8,'mean_below_threshold'),('min_score',.6,'minimum_below_threshold')]:
            changed=dict(requirement,**{key:value});result=client.check_callback_evaluation_summary('default',[changed]);self.assertFalse(result['passed']);self.assertIn(reason,result['checks'][0]['reasons'])
        calls=client.calls
        for requirements in [[],[requirement,requirement],[dict(requirement,min_count=True)],[dict(requirement,evaluator_version=True)],[dict(requirement,min_mean=float('nan'))],[dict(requirement,min_mean=10**1000)],[dict(requirement,unexpected=1)],[dict(evaluator_id='check',evaluator_version=2,metric='quality',min_count=1)]]:
            with self.assertRaises(ValueError):client.check_callback_evaluation_summary('default',requirements)
        with self.assertRaises(ValueError):client.check_callback_evaluation_summary('default',[requirement],max_failed_assessments=True)
        self.assertEqual(client.calls,calls)

    def test_callback_version_failure_gate(self):
        packet=dict(kind='callback_evaluation_summary',project_id='default',since_ms=None,until_ms=None,assessment_source='caller_reported',provider_calls=0,automatic_promotion=False,trace_count=3,selected_tasks=3,skipped_tasks=0,completed_assessments=1,failed_assessments=2,metrics=[dict(evaluator_id='check',evaluator_version=2,metric='quality',count=1,mean=1,min=1,max=1)],evaluators=[dict(evaluator_id='check',evaluator_version=2,completed_assessments=1,failed_assessments=0),dict(evaluator_id='check',evaluator_version=3,completed_assessments=0,failed_assessments=2)])
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return packet
        client=Fake('http://127.0.0.1:1');requirement=dict(evaluator_id='check',evaluator_version=2,metric='quality',min_count=1,min_mean=1,max_failed_assessments=0)
        result=client.check_callback_evaluation_summary('default',[requirement]);self.assertTrue(result['passed']);self.assertEqual(result['checks'][0]['evaluator_attempts']['failed_assessments'],0)
        packet['evaluators'][0]['failed_assessments']=1;packet['evaluators'][1]['failed_assessments']=1
        result=client.check_callback_evaluation_summary('default',[requirement]);self.assertFalse(result['passed']);self.assertIn('evaluator_failure_limit_exceeded',result['checks'][0]['reasons'])
        result=client.check_callback_evaluation_summary('default',[dict(requirement,max_failed_assessments=1)]);self.assertTrue(result['passed'])
        del packet['evaluators'];result=client.check_callback_evaluation_summary('default',[requirement]);self.assertFalse(result['passed']);self.assertIn('evaluator_attempts_unavailable',result['checks'][0]['reasons'])
        with self.assertRaises(ValueError):client.check_callback_evaluation_summary('default',[dict(requirement,max_failed_assessments=True)])

    def test_evaluator_identity_validation(self):
        client=Studio('http://127.0.0.1:1')
        for identity in [dict(evaluator_id='metric'),dict(evaluator_version=1),dict(evaluator_id='private text',evaluator_version=1),dict(evaluator_id='metric',evaluator_version=True),dict(evaluator_id='metric',evaluator_version=0)]:
            with self.assertRaises(ValueError):client.track('task',evaluate=lambda _: {'metric':1},**identity)
        with self.assertRaises(ValueError):client.track('task',evaluator_id='metric',evaluator_version=1)

    def test_online_evaluation_sampling(self):
        class Fake(Studio):
            def __init__(self):pass
            def request(self,path,body=None,timeout=10):return {'id':'trace-one'}
        client=Fake();calls=[]
        def evaluate(result):calls.append(result);return {'quality':1}
        @client.track('zero',evaluate=evaluate,evaluation_sample_rate=0)
        def zero():return 1
        with client.trace('default','zero') as trace:self.assertEqual(zero(),1)
        self.assertEqual(calls,[]);self.assertEqual(len(trace.spans),2);self.assertEqual(trace.spans[1]["usage"]["evaluation_sampling"],dict(method="sha256_v1",sample_rate=0,selected=False))
        @client.track('sample',evaluate=evaluate,evaluation_sample_rate=0.5)
        def task():return 'value'
        import asyncio
        @client.track('async-zero',evaluate=evaluate,evaluation_sample_rate=0)
        async def async_zero():return 2
        async def run_zero():
            with client.trace('default','async-zero') as async_trace:self.assertEqual(await async_zero(),2)
            self.assertEqual(len(async_trace.spans),2)
        asyncio.run(run_zero());self.assertEqual(calls,[])
        decisions=[]
        for i in range(100):
            before=len(calls)
            with client.trace('default','sample-'+str(i)):task()
            decisions.append(len(calls)>before)
        self.assertTrue(any(decisions));self.assertTrue(not all(decisions))
        for i in range(100):
            before=len(calls)
            with client.trace('default','sample-'+str(i)):task()
            self.assertEqual(len(calls)>before,decisions[i])
        for rate in [True,-1,1.1,float('nan'),float('inf'),'0.5']:
            with self.assertRaises(ValueError):client.track('bad',evaluate=evaluate,evaluation_sample_rate=rate)
        with self.assertRaises(ValueError):client.track('bad',evaluation_sample_rate=0)

    def test_online_track_evaluation(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):self.packets=[]
            def request(self,path,body=None,timeout=10):self.packets.append(body);return {'id':'trace-one'}
        client=Fake();seen=[]
        @client.track('task',evaluate=lambda result:(seen.append(result) or {'accuracy':1.0}),evaluator_id='accuracy-check',evaluator_version=2)
        def task():return 'PRIVATE_OUTPUT'
        self.assertEqual(task(),'PRIVATE_OUTPUT');self.assertEqual(seen,[])
        with client.trace('default','eval') as trace:self.assertEqual(task(),'PRIVATE_OUTPUT')
        self.assertEqual(seen,['PRIVATE_OUTPUT']);self.assertEqual(trace.spans[2]['usage']['evaluation_scores'],{'accuracy':1.0});self.assertEqual(trace.spans[2]['usage']['evaluator_ref'],{'id':'accuracy-check','version':2});self.assertEqual(trace.spans[2]['parent_id'],1);self.assertNotIn('PRIVATE_OUTPUT',str(client.packets))
        @client.track('bad',evaluate=lambda _: {'invalid':float('nan')})
        def bad():return 42
        with client.trace('default','bad-eval') as failed:self.assertEqual(bad(),42)
        self.assertEqual(failed.spans[1]['status'],'completed');self.assertEqual(failed.spans[2]['status'],'failed');self.assertEqual(len(failed.evaluation_errors),1);self.assertNotIn('evaluation_scores',failed.spans[2]['usage'])
        @client.track('business-error',evaluate=lambda result:seen.append(result))
        def business_error():raise RuntimeError('PRIVATE_BUSINESS_ERROR')
        count=len(seen)
        with self.assertRaises(RuntimeError):
            with client.trace('default','business-failure') as business_trace:business_error()
        self.assertEqual(len(seen),count);self.assertEqual(len(business_trace.spans),2);self.assertNotIn('PRIVATE_BUSINESS_ERROR',str(client.packets))
        async def evaluate(result):return {'quality':0.5}
        @client.track('async-task',evaluate=evaluate)
        async def async_task():return 'private'
        async def run():
            with client.trace('default','async-eval') as trace:
                self.assertEqual(await async_task(),'private')
            self.assertEqual(trace.spans[2]['usage']['evaluation_scores'],{'quality':0.5})
        asyncio.run(run())
        with self.assertRaises(ValueError):client.track('wrong',evaluate=evaluate)(lambda:1)

    def test_memory_expiry(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.memory_expiry('default',include_global=True,horizon_days=7)
        self.assertEqual(client.calls,['memory/expiry?project_id=default&include_global=true&horizon_days=7'])
        for options in [dict(include_global=1),dict(horizon_days=True),dict(horizon_days=-1),dict(horizon_days=366)]:
            with self.assertRaises(ValueError):client.memory_expiry('default',**options)
        self.assertEqual(len(client.calls),1)

    def test_review_queue_history(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.review_queue_history('queue-one',offset=20,limit=10)
        self.assertEqual(client.calls,['observability/review-queues/queue-one/history?offset=20&limit=10'])
        for bounds in [dict(offset=True),dict(offset=2001),dict(limit=0),dict(limit=101)]:
            with self.assertRaises(ValueError):client.review_queue_history('queue-one',**bounds)
        self.assertEqual(len(client.calls),1)

    def test_review_queue_archive(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.set_review_queue_archived('queue-one',True,base_version=3)
        self.assertEqual(client.calls[0],('observability/review-queues/queue-one/lifecycle',dict(base_version=3,archived=True)))
        client.review_queues('default',archived=False);self.assertIn('archived=false',client.calls[1][0])
        for archived,version in [(1,1),(True,True),(False,0)]:
            with self.assertRaises(ValueError):client.set_review_queue_archived('queue-one',archived,base_version=version)
        with self.assertRaises(ValueError):client.review_queues('default',archived=1)
        self.assertEqual(len(client.calls),2)

    def test_review_queue_export_csv(self):
        import copy,csv,io
        packet=dict(id='queue-one',version=4,project_id='default',name='=SUM(1)',targets=[dict(trace_id='trace-one',span_id=1),dict(trace_id='trace-two')],assignments={'0':'Reviewer'},completions={'0':dict(reviewer='Reviewer',feedback_version=2,annotation_id='review-one')})
        class Fake(Studio):
            def __init__(self):self.packet=copy.deepcopy(packet);self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return self.packet
        client=Fake();rows=list(csv.DictReader(io.StringIO(client.export_review_queue_csv('queue-one'))))
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['queue_name'],"'=SUM(1)");self.assertEqual(rows[0]['feedback_version'],'2');self.assertEqual(rows[0]['annotation_id'],'review-one');self.assertEqual(rows[1]['status'],'unassigned');self.assertEqual(rows[1]['span_id'],'');self.assertEqual(len(client.calls),1)
        for mutation in [lambda q:q.update(id='other'),lambda q:q['completions']['0'].update(reviewer='Other'),lambda q:q['targets'].append(q['targets'][0]),lambda q:q['assignments'].update({'9':'Unknown'}),lambda q:q['targets'][0].update(span_id=True)]:
            client.packet=copy.deepcopy(packet);mutation(client.packet)
            with self.assertRaises(EvaluationError):client.export_review_queue_csv('queue-one')

    def test_review_queue_filters(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.review_queues('default',status='pending',reviewer='Имя + имя',name='ПРОВЕРКА')
        import urllib.parse
        query=urllib.parse.parse_qs(urllib.parse.urlsplit(client.calls[0]).query)
        self.assertEqual(query['name'],['ПРОВЕРКА']);self.assertEqual(query['status'],['pending']);self.assertEqual(query['reviewer'],['Имя + имя'])
        for filters in [dict(name=' '),dict(name=1),dict(name='я'*101),dict(status='unknown'),dict(reviewer=' '),dict(reviewer=1),dict(reviewer='я'*101)]:
            with self.assertRaises(ValueError):client.review_queues('default',**filters)
        self.assertEqual(len(client.calls),1)

    def test_review_queue_sources_and_pages(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();targets=[dict(trace_id='trace-one',span_id=1)]
        client.create_review_queue('default','Review',targets,instructions='Check accuracy');targets[0]['span_id']=2
        self.assertEqual(client.calls[0][1]['targets'][0]['span_id'],1)
        client.review_queue('queue-one');client.review_queues('default',offset=20,limit=10)
        self.assertEqual(client.calls[2][0],'observability/review-queues?project_id=default&offset=20&limit=10')
        for targets in [[],[dict(trace_id='trace-one')]*2,[dict(trace_id='trace-one',span_id=True)],[dict(trace_id='trace-one',extra=1)]]:
            with self.assertRaises(ValueError):client.create_review_queue('default','Review',targets)
        for bounds in [dict(offset=True),dict(limit=101),dict(offset=1001)]:
            with self.assertRaises(ValueError):client.review_queues('default',**bounds)
        self.assertEqual(len(client.calls),3)

    def test_feedback_read_write_capture_and_validation(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body));return {}
        client=Fake();client.feedback('trace-test');client.feedback('trace-test',version=0)
        self.assertEqual(client.calls[1][0],'observability/traces/trace-test/feedback?version=0')
        annotation=dict(author='Reviewer',metric='quality',value=0,comment='original')
        client.save_feedback('trace-test',annotation,base_version=0);annotation['comment']='changed'
        self.assertEqual(client.calls[2][1]['annotation']['comment'],'original')
        for bad in [dict(author=''),dict(author='x',value=True,metric='quality'),dict(author='x',value=10**400,metric='quality'),dict(author='x',metric='quality',value=1,category='good'),dict(author='x',comment='ok',deleted=1),dict(author='x',comment='ok',span_id=True),dict(author='x',comment='я'*8001),dict(author='x',comment='ok',unknown=1)]:
            with self.assertRaises(ValueError):client.save_feedback('trace-test',bad,base_version=0)
        for version in [-1,True,2001]:
            with self.assertRaises(ValueError):client.feedback('trace-test',version=version)
        self.assertEqual(len(client.calls),3)

    def test_feedback_history_pages_validate_before_request(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {}
        client=Fake();client.feedback_versions('trace-test',offset=20,limit=10)
        self.assertEqual(client.calls,['observability/traces/trace-test/feedback/versions?offset=20&limit=10'])
        for bounds in [dict(offset=-1),dict(offset=True),dict(offset=2001),dict(limit=0),dict(limit=101),dict(limit=True)]:
            with self.assertRaises(ValueError):client.feedback_versions('trace-test',**bounds)
        self.assertEqual(len(client.calls),1)

    def test_unscored_playground_csv_export(self):
        import csv,io,copy
        packet=dict(kind='experiment_export',schema_version=1,run_id='run',outputs_included=False,
            dataset_id='input',dataset_version=1,dataset_sha256='hash',status='completed',strict_quality=False,
            playground=True,metrics=[],items=[dict(sample_id='input',status='completed',duration_ms=7,
                has_error=False,output_truncated=False,scores={},output='answer')])
        class Fake(Studio):
            def export_experiment(self,run_id,*,include_outputs=False):
                result=copy.deepcopy(packet);result['outputs_included']=include_outputs;return result
        client=Fake()
        rows=list(csv.DictReader(io.StringIO(client.export_experiment_csv('run'))))
        self.assertEqual(len(rows),1);self.assertNotIn('output',rows[0])
        self.assertFalse(any(key.startswith('score_') for key in rows[0]))
        self.assertEqual(list(csv.DictReader(io.StringIO(client.export_experiment_csv('run',include_outputs=True))))[0]['output'],'answer')
        packet['strict_quality']=True
        with self.assertRaises(EvaluationError):client.export_experiment_csv('run')
        packet['strict_quality']=False;packet['playground']=False
        with self.assertRaises(EvaluationError):client.export_experiment_csv('run')

    def test_experiment_export_requires_explicit_answer_opt_in(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append(path);return {'ok':True}
        client=Fake()
        client.export_experiment('run')
        self.assertEqual(client.calls[-1],'evaluation/experiments/run/export?include_outputs=false')
        client.export_experiment('run',include_outputs=True)
        self.assertEqual(client.calls[-1],'evaluation/experiments/run/export?include_outputs=true')
        for value in [1,'true',None]:
            with self.assertRaises(ValueError):client.export_experiment('run',include_outputs=value)
        with self.assertRaises(ValueError):client.export_experiment('../escape')
        self.assertEqual(len(client.calls),2)

    def test_background_export_output_opt_in(self):
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return (path,body)
        client=Fake();self.assertEqual(client.export_background('session'),('sessions/session/background',{'action':'export','include_outputs':False}))
        self.assertTrue(client.export_background('session',include_outputs=True)[1]['include_outputs'])
        for value in [1,'true',None]:
            with self.assertRaises(ValueError):client.export_background('session',include_outputs=value)

    def test_background_name_utf8_bounds(self):
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return body
        self.assertEqual(Fake().start_background('session','true',name='Сборка')['name'],'Сборка')
        for name in ['',True,'я'*101]:
            with self.assertRaises(ValueError):Fake().start_background('session','true',name=name)

    def test_background_routes_bounds_and_wait_transport_deadline(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,timeout=10):self.calls.append((path,body,timeout));return {'ok':True}
        client=Fake()
        client.cleanup_background('session','bg-1');self.assertEqual(client.calls[-1][1],{'action':'cleanup','task_id':'bg-1'})
        client.start_background('session','printf done',timeout=12)
        self.assertEqual(client.calls[-1],('sessions/session/background',{'action':'start','command':'printf done','timeout':12},10))
        client.start_background('session','printf done',follow_up='Inspect output.')
        self.assertEqual(client.calls[-1][1]['follow_up'],'Inspect output.')
        client.wait_background('session',['bg-1'],wait_seconds=60)
        self.assertEqual(client.calls[-1][2],65)
        client.background_tasks('session');client.background_output('session','bg-1');client.cancel_background('session','bg-1');client.cleanup_background('session')
        before=len(client.calls)
        for command,timeout in [('',1),('x',True),('x',0),('x',86401),('я'*8001,1)]:
            with self.assertRaises(ValueError):client.start_background('session',command,timeout=timeout)
        for ids,wait in [([],1),(['bg-1']*2,1),(['bg-1'],True),(['bg-1'],61),(['../bad'],1)]:
            with self.assertRaises(ValueError):client.wait_background('session',ids,wait_seconds=wait)
        for follow_up in ['',True,'я'*8001]:
            with self.assertRaises(ValueError):client.start_background('session','printf done',follow_up=follow_up)
        self.assertEqual(len(client.calls),before)

    def test_stream_first_text_measurement_ignores_empty_and_control_events(self):
        class Fake(Studio):
            def __init__(self):self.packets=[]
            def ingest_trace(self,*args,**kwargs):self.packets.append(args);return {'id':'saved'}
        class Stream:
            def __init__(self,chunks):self.chunks=iter(chunks)
            def __iter__(self):return self
            def __next__(self):return next(self.chunks)
            def close(self):pass
        client=Fake()
        variants=[(client.track_openai_chat_stream,[{'choices':[{'delta':{'role':'assistant'}}]},{'choices':[{'delta':{'content':'PRIVATE TEXT'}}]}]),
                  (client.track_openai_responses_stream,[{'type':'response.created'},{'type':'response.output_text.delta','delta':'PRIVATE TEXT'}]),
                  (client.track_anthropic_messages_stream,[{'type':'content_block_delta','delta':{'type':'text_delta','text':''}},{'type':'content_block_delta','delta':{'type':'text_delta','text':'PRIVATE TEXT'}}])]
        for adapter,chunks in variants:
            with client.trace('default','first-text') as trace:
                wrapped=adapter(lambda **kwargs:Stream(chunks),provider_id='local',model_id='model')
                stream=wrapped(model='model',stream=True)
                span=stream._span
                trace._elapsed=lambda:span.start+17
                self.assertIs(next(stream),chunks[0]);self.assertNotIn('first_text_ms',trace.spans[span.index]['usage'])
                self.assertIs(next(stream),chunks[1]);self.assertEqual(trace.spans[span.index]['usage']['first_text_ms'],17)
                trace._elapsed=lambda:span.start+25
                stream.close()
            self.assertEqual(client.packets[-1][3][1]['usage']['first_text_ms'],17)
            self.assertNotIn('PRIVATE',str(client.packets[-1]))

    def test_async_stream_first_text_is_measured_once(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):self.packets=[]
            def ingest_trace(self,*args,**kwargs):self.packets.append(args);return {'id':'saved'}
        class Stream:
            def __init__(self):self.chunks=iter([{'choices':[]},{'choices':[{'delta':{'content':'PRIVATE'}}]},{'choices':[{'delta':{'content':'MORE PRIVATE'}}]}])
            def __aiter__(self):return self
            async def __anext__(self):
                try:return next(self.chunks)
                except StopIteration:raise StopAsyncIteration
            async def close(self):pass
        async def exercise():
            client=Fake()
            async def create(**kwargs):return Stream()
            with client.trace('default','async-first-text') as trace:
                stream=await client.track_openai_chat_stream(create,provider_id='local',model_id='model')(model='model',stream=True)
                span=stream._span;trace._elapsed=lambda:span.start+11
                await stream.__anext__();self.assertNotIn('first_text_ms',trace.spans[1]['usage'])
                await stream.__anext__();trace._elapsed=lambda:span.start+30
                await stream.__anext__();await stream.close()
            self.assertEqual(client.packets[0][3][1]['usage']['first_text_ms'],11)
            self.assertNotIn('PRIVATE',str(client.packets))
        asyncio.run(exercise())

    def test_trace_time_series_boundaries_and_no_request_on_invalid_range(self):
        class Fake(Studio):
            def __init__(self):self.paths=[]
            def request(self,path):self.paths.append(path);return {}
        client=Fake();client.trace_time_series(project_id='default',since_ms=0,until_ms=499,bucket_ms=1)
        self.assertIn('observability/time-series?',client.paths[0]);self.assertIn('since_ms=0',client.paths[0]);self.assertIn('project_id=default',client.paths[0])
        for options in (dict(since_ms=0,until_ms=500,bucket_ms=1),dict(since_ms=0,until_ms=10,bucket_ms=0),dict(since_ms=11,until_ms=10,bucket_ms=1),dict(since_ms=False,until_ms=10,bucket_ms=1),dict(since_ms=0,until_ms=2**64-1,bucket_ms=86400000)):
            with self.assertRaises(ValueError):client.trace_time_series(**options)
        self.assertEqual(len(client.paths),1)
        client.trace_time_series(since_ms=0,until_ms=10,bucket_ms=10,status='failed');self.assertIn('status=failed',client.paths[-1])
        for status in ['',True,'FAILED','bad-status','x'*41]:
            with self.assertRaises(ValueError):client.trace_time_series(since_ms=0,until_ms=10,bucket_ms=10,status=status)
        self.assertEqual(len(client.paths),2)

    def test_trace_status_and_time_filter_admission(self):
        class Fake(Studio):
            def __init__(self):self.paths=[]
            def request(self,path):self.paths.append(path);return {}
        client=Fake();client.traces(status='failed',since_ms=0,until_ms=100,guardrail='blocked')
        self.assertIn('status=failed',client.paths[0]);self.assertIn('since_ms=0',client.paths[0]);self.assertIn('until_ms=100',client.paths[0])
        for kwargs in ({'status':'PRIVATE STATUS'},{'since_ms':True},{'since_ms':-1},{'until_ms':2**64},{'since_ms':101,'until_ms':100}):
            with self.assertRaises(ValueError):client.traces(**kwargs)
        self.assertEqual(len(client.paths),1)

    def test_guardrail_trace_catalog_filters_and_bounds(self):
        import urllib.parse
        class Fake(Studio):
            def __init__(self):self.paths=[]
            def request(self,path):self.paths.append(path);return {'traces':[]}
        client=Fake();client.traces(project_id='project',guardrail='blocked',offset=20,limit=20)
        parsed=urllib.parse.parse_qs(client.paths[0].split('?',1)[1])
        self.assertEqual(parsed,dict(project_id=['project'],guardrail=['blocked'],offset=['20'],limit=['20'],removed=['false']))
        for kwargs in ({'guardrail':'unknown'},{'offset':True},{'limit':201},{'removed':'false'},{'guardrail_policy_sha256':'A'*64},{'guardrail_policy_sha256':42}):
            with self.assertRaises(ValueError):client.traces(**kwargs)
        self.assertEqual(len(client.paths),1)
        client.traces(guardrail_policy_sha256='a'*64,guardrail='blocked')
        self.assertIn('guardrail_policy_sha256='+'a'*64,client.paths[-1])

    def test_review_combined_results_only_reads_matching_saved_answers(self):
        class Fake(Studio):
            def __init__(self):
                self.paths=[]
                pins=dict(project_id='default',dataset_id='data',dataset_version=1,dataset_sha256='hash')
                self.score=dict(pins,id='score',kind='offline_score',metrics=['exact_match'],mean_scores={'exact_match':0.5},items=[dict(sample_id='a',output='answer')])
                self.run=dict(pins,id='judge',status='completed',plan_id='plan',plan_sha256='planhash',mean_score=0.75)
                self.plan=dict(pins,id='plan',kind='judge_plan',plan_sha256='planhash',offline_score_source={'id':'score'},samples=[dict(sample_id='a',output='answer')])
            def request(self,path,*args,**kwargs):
                self.paths.append(path)
                if args or set(kwargs)-{'timeout'}:raise AssertionError('Review must use GET only')
                return {'evaluation/score/score':self.score,'evaluation/judge-runs/judge':self.run,'evaluation/judge-plans/plan':self.plan}[path]
        client=Fake()
        result=client.review_scored_output_judges('score','judge',min_scores={'exact_match':0.5},min_judge_score=0.75)
        self.assertTrue(result['passed']);self.assertEqual(result['provider_calls'],0)
        self.assertFalse(result['automatic_promotion']);self.assertEqual(len(client.paths),3)
        self.assertFalse(client.review_scored_output_judges('score','judge',min_judge_score=0.8)['passed'])
        self.assertFalse(client.review_scored_output_judges('score','judge',min_scores={'exact_match':1})['passed'])
        for field in ('project_id','dataset_id','dataset_version','dataset_sha256','plan_sha256'):
            original=client.run[field];client.run[field]='other'
            with self.assertRaises(EvaluationError):client.review_scored_output_judges('score','judge')
            client.run[field]=original
        client.plan['offline_score_source']['id']='different'
        with self.assertRaises(EvaluationError):client.review_scored_output_judges('score','judge')
        client.plan['offline_score_source']['id']='score'
        client.plan['samples'][0]['output']='substituted'
        with self.assertRaises(EvaluationError):client.review_scored_output_judges('score','judge')
        client.plan['samples'][0]['output']='answer';client.run['status']='running'
        with self.assertRaises(EvaluationError):client.review_scored_output_judges('score','judge')
        before=len(client.paths)
        with self.assertRaises(ValueError):client.review_scored_output_judges('score','judge',min_judge_score=float('nan'))
        self.assertEqual(len(client.paths),before)

    def test_combined_gate_requires_both_results_and_keeps_partial_evidence(self):
        class Fake(Studio):
            def __init__(self):self.judge_pass=True;self.failure=None;self.judge_calls=0
            def scored_outputs(self,id):return dict(id=id,kind='offline_score',project_id='default',metrics=['exact_match'],mean_scores={'exact_match':0.5})
            def plan_scored_output_judges(self,*args,**kwargs):return {'id':'plan'}
            def judge_run(self,id):return {'id':id,'status':'interrupted','items':[]}
            def evaluate_judge_plan(self,*args,**kwargs):
                self.judge_calls+=1
                if self.failure:raise self.failure
                return {'passed':self.judge_pass,'run':{'id':'judge'}}
        client=Fake();settings={'project_id':'default'}
        result=client.evaluate_scored_output_judges('score',settings,'rubric',min_scores={'exact_match':1})
        self.assertFalse(result['passed']);self.assertTrue(result['judge']['passed']);self.assertEqual(client.judge_calls,1)
        client.judge_pass=False
        result=client.evaluate_scored_output_judges('score',settings,'rubric',min_scores={'exact_match':0})
        self.assertFalse(result['passed']);self.assertTrue(result['deterministic']['passed'])
        client.judge_pass=True
        self.assertTrue(client.evaluate_scored_output_judges('score',settings,'rubric',min_scores={'exact_match':0})['passed'])
        original=EvaluationError('judge_timeout','judge');client.failure=original
        with self.assertRaises(EvaluationError) as raised:client.evaluate_scored_output_judges('score',settings,'rubric')
        self.assertIs(raised.exception,original);self.assertEqual(original.judge_plan_id,'plan')
        self.assertEqual(original.deterministic_result['receipt']['id'],'score')
        self.assertEqual(original.judge_run_receipt['status'],'interrupted')
        self.assertFalse(original.judge_evidence_unavailable)
        def unavailable(id):raise EvaluationError('read_failed')
        client.judge_run=unavailable
        with self.assertRaises(EvaluationError) as missing:client.evaluate_scored_output_judges('score',settings,'rubric')
        self.assertIs(missing.exception,original)
        self.assertTrue(original.judge_evidence_unavailable)
        self.assertIsNone(original.judge_run_receipt)
        before=client.judge_calls
        with self.assertRaises(ValueError):client.evaluate_scored_output_judges('score',settings,'rubric',min_scores={'missing':1})
        self.assertEqual(client.judge_calls,before)

    def test_plan_scored_answers_pins_verified_source(self):
        class Fake(Studio):
            def scored_outputs(self,id):return dict(id=id,kind='offline_score',project_id='default',dataset_id='data',dataset_version=2,items=[dict(sample_id='a',output='ready')])
            def request(self,path,body):return body
        client=Fake();settings={'project_id':'default'}
        plan=client.plan_scored_output_judges('score',settings,'rubric')
        self.assertEqual(plan['offline_score_id'],'score');self.assertEqual(plan['outputs'],{'a':'ready'})
        self.assertEqual(plan['dataset_version'],2)
        with self.assertRaises(EvaluationError):client.plan_scored_output_judges('score',{'project_id':'other'},'rubric')
        with self.assertRaises(ValueError):client.plan_judges('data',2,{'a':'ready'},settings,'rubric',experiment_id='run',offline_score_id='score')

    def test_scored_outputs_catalog_encodes_pagination(self):
        class Fake(Studio):
            def request(self,path):return path
        client=Fake()
        self.assertEqual(client.list_scored_outputs('default',dataset_id='data',offset=20,limit=10),'evaluation/score?project_id=default&offset=20&limit=10&dataset_id=data')
        with self.assertRaises(ValueError):client.list_scored_outputs('default',limit=0)
        with self.assertRaises(ValueError):client.list_scored_outputs('default',offset=True)

    def test_ci_preserves_score_result_when_trace_delivery_fails(self):
        import importlib.util,tempfile,pathlib,json,contextlib,io
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('evaluation_cli',pathlib.Path(__file__).with_name('evaluate-studio.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class Fake(Studio):
            def evaluate_task(self,*args,**kwargs):return dict(passed=True,receipt={'id':'score-completed'},comparison=None)
            def ingest_trace(self,*args,**kwargs):raise EvaluationError('connection_error')
        with tempfile.TemporaryDirectory() as directory:
            source=pathlib.Path(directory)/'task.py';source.write_text("def task(text):return text\n")
            report=pathlib.Path(directory)/'report.json'
            argv=['evaluate-studio.py','--base-url','http://127.0.0.1:8100','--dataset-id','data','--dataset-version','1','--metric','exact_match','--task-file',str(source),'--trace-correlation','test','--report',str(report)]
            with patch.object(module,'Studio',Fake),patch.object(module.sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(),2)
            result=json.loads(report.read_text())
            self.assertTrue(result['evaluation_result']['passed'])
            self.assertEqual(result['evaluation_result']['receipt']['id'],'score-completed')
            self.assertTrue(result['trace_export_failed']);self.assertIsNone(result['trace_id'])

    def test_task_trace_samples_capture_only_metadata(self):
        import asyncio,json
        class Fake(Studio):
            def __init__(self):self.sent=[]
            def metrics(self):return dict(max_metrics_per_run=6,metrics=[dict(id='exact_match',reference_requirement='required')])
            def request(self,path):return dict(id='data',project_id='default',version=1,sha256='hash',samples=[dict(id='a',input='PRIVATE_INPUT',expected_output='PRIVATE_REFERENCE'),dict(id='b',input='PRIVATE_INPUT_2',expected_output='PRIVATE_REFERENCE')])
            def evaluate_outputs(self,*args,**kwargs):return {'passed':True}
            def ingest_trace(self,*args):self.sent.append(args);return {'id':'trace-task'}
        client=Fake()
        async def run():
            async def task(text):await asyncio.sleep(0);return 'PRIVATE_OUTPUT'
            with client.trace('default','task-test') as trace:
                result=await client.evaluate_task_async('data',1,task,['exact_match'],concurrency=2,trace=trace)
                self.assertTrue(result['passed'])
            self.assertEqual([span['name'] for span in trace.spans],['pipeline','sample.a','sample.b'])
            self.assertEqual([span['parent_id'] for span in trace.spans],[None,0,0])
            self.assertTrue(all(span['status']=='completed' for span in trace.spans))
            return trace
        trace=asyncio.run(run())
        self.assertNotIn('PRIVATE_',json.dumps(client.sent))
        calls=[]
        with client.trace('foreign','foreign-project') as foreign:
            with self.assertRaises(ValueError):client.evaluate_task('data',1,lambda text:calls.append(text) or 'ok',['exact_match'],trace=foreign)
        self.assertEqual(calls,[])
        with self.assertRaises(ValueError):client.evaluate_task('data',1,lambda _: 'ok',['exact_match'],trace=trace)
        original=RuntimeError('PRIVATE_ERROR')
        def fail(text):raise original
        with self.assertRaises(RuntimeError):
            with client.trace('default','failure') as failed:
                client.evaluate_task('data',1,fail,['exact_match'],trace=failed)
        self.assertEqual(failed.spans[1]['status'],'failed')
        self.assertEqual(failed.spans[0]['status'],'failed')
        self.assertNotIn('PRIVATE_',json.dumps(client.sent))

    def test_async_task_bounded_parallelism_and_failure_cleanup(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):self.scored=[]
            def metrics(self):return dict(max_metrics_per_run=6,metrics=[dict(id='exact_match',reference_requirement='required')])
            def request(self,path):return dict(id='data',project_id='default',version=1,sha256='hash',samples=[dict(id=str(i),input=str(i),expected_output='ok') for i in range(6)])
            def evaluate_outputs(self,*args,**kwargs):self.scored.append(args[2]);return args[2]
        async def exercise():
            client=Fake();active=0;peak=0;calls=[];ready=asyncio.Event()
            async def parallel(text):
                nonlocal active,peak
                calls.append(text);active+=1;peak=max(peak,active)
                if active==3:ready.set()
                try:await ready.wait();await asyncio.sleep(0);return 'ok'
                finally:active-=1
            result=await client.evaluate_task_async('data',1,parallel,['exact_match'],concurrency=3)
            self.assertEqual(peak,3);self.assertEqual(len(calls),6);self.assertEqual(list(result),list('012345'))
            client.scored.clear();calls.clear();cleanup=[];ready=asyncio.Event();original=RuntimeError('failed')
            async def failing(text):
                calls.append(text)
                if len(calls)==2:ready.set()
                try:
                    await ready.wait()
                    if text=='0':raise original
                    await asyncio.Event().wait()
                finally:cleanup.append(text)
            with self.assertRaises(RuntimeError) as raised:
                await client.evaluate_task_async('data',1,failing,['exact_match'],concurrency=2)
            self.assertIs(raised.exception,original)
            self.assertEqual(set(cleanup),{'0','1'});self.assertEqual(calls,['0','1'])
            self.assertEqual(original.started_sample_ids,['0','1']);self.assertEqual(original.completed_outputs,{})
            self.assertEqual(client.scored,[])
            with self.assertRaises(ValueError):await client.evaluate_task_async('data',1,parallel,['exact_match'],concurrency=9)
        asyncio.run(exercise())

    def test_task_contexts_are_explicit_frozen_and_reference_free(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):self.samples=[dict(id='a',input='question',expected_output='PRIVATE_REFERENCE',metadata={'private':'secret'},contexts=['source one','source two'])]
            def metrics(self):return dict(max_metrics_per_run=6,metrics=[dict(id='exact_match',reference_requirement='required')])
            def request(self,path):return dict(id='data',project_id='default',version=1,sha256='hash',samples=self.samples)
            def evaluate_outputs(self,*args,**kwargs):return args[2]
        client=Fake();seen=[]
        def rag(text,contexts):
            seen.append((text,contexts))
            self.assertIsInstance(contexts,tuple)
            client.samples[0]['contexts'].append('later mutation')
            return 'answer'
        self.assertEqual(client.evaluate_task('data',1,rag,['exact_match'],with_contexts=True),{'a':'answer'})
        self.assertEqual(seen,[('question',('source one','source two'))])
        self.assertNotIn('PRIVATE_REFERENCE',str(seen));self.assertNotIn('secret',str(seen))
        async def async_rag(text,contexts):return text+':'+str(len(contexts))
        result=asyncio.run(client.evaluate_task_async('data',1,async_rag,['exact_match'],with_contexts=True))
        self.assertEqual(result,{'a':'question:3'})
        client.samples[0]['contexts']=['ok',123]
        with self.assertRaises(EvaluationError):client.evaluate_task('data',1,rag,['exact_match'],with_contexts=True)
        self.assertEqual(len(seen),1)
        with self.assertRaises(ValueError):client.evaluate_task('data',1,rag,['exact_match'],with_contexts='yes')

    def test_async_task_evaluation_timeout_cancellation_and_partial_outputs(self):
        import asyncio,threading
        class Fake(Studio):
            def __init__(self):self.scored=[];self.threads=[]
            def metrics(self):
                self.threads.append(threading.get_ident())
                return dict(max_metrics_per_run=6,metrics=[dict(id='exact_match',reference_requirement='required')])
            def request(self,path):return dict(id='data',project_id='default',version=1,sha256='hash',samples=[dict(id='a',input='one',expected_output='ok'),dict(id='b',input='two',expected_output='ok')])
            def evaluate_outputs(self,*args,**kwargs):
                self.threads.append(threading.get_ident());self.scored.append(args[2]);return {'passed':True}
        async def exercise():
            client=Fake();called=[];loop_thread=threading.get_ident()
            async def good(text):called.append(text);await asyncio.sleep(0);return 'ok'
            result=await client.evaluate_task_async('data',1,good,['exact_match'])
            self.assertTrue(result['passed']);self.assertEqual(called,['one','two'])
            self.assertTrue(all(thread!=loop_thread for thread in client.threads))
            client.scored.clear();called.clear();cancelled=[]
            async def slow(text):
                called.append(text)
                if text=='one':return 'first'
                try:await asyncio.Event().wait()
                finally:cancelled.append(text)
            with self.assertRaises(asyncio.TimeoutError) as raised:
                await client.evaluate_task_async('data',1,slow,['exact_match'],item_timeout=0.01)
            self.assertEqual(raised.exception.completed_outputs,{'a':'first'})
            self.assertEqual(raised.exception.sample_id,'b');self.assertEqual(cancelled,['two'])
            self.assertEqual(client.scored,[])
            original=asyncio.CancelledError()
            async def cancel(text):raise original
            with self.assertRaises(asyncio.CancelledError) as raised:
                await client.evaluate_task_async('data',1,cancel,['exact_match'])
            self.assertEqual(raised.exception.completed_outputs,{})
            self.assertEqual(raised.exception.sample_id,'a')
            with self.assertRaises(ValueError):await client.evaluate_task_async('data',1,good,['exact_match'],item_timeout=0)
        asyncio.run(exercise())

    def test_task_junit_separates_samples_gate_and_missing_scores(self):
        import importlib.util
        import xml.etree.ElementTree as ET
        spec=importlib.util.spec_from_file_location('evaluation_junit',pathlib.Path(__file__).with_name('evaluate-studio.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        evaluated=dict(passed=False,receipt=dict(id='score',mean_scores={'exact_match':0.5},items=[dict(sample_id='a',output='PRIVATE_ANSWER',scores={'exact_match':1}),dict(sample_id='b',output='PRIVATE_ANSWER',scores={'exact_match':0})]),thresholds={'exact_match':1})
        suite=module.task_junit(evaluated,{},1)
        self.assertEqual(suite.attrib,dict(name='allpaka-python-task',tests='3',failures='1',errors='0',skipped='0'))
        self.assertEqual([case.get('name') for case in suite.findall('testcase')],['a','b','quality_gate'])
        self.assertTrue(all(case.find('failure') is None for case in suite.findall('testcase')[:2]))
        self.assertNotIn('PRIVATE_ANSWER',ET.tostring(suite,encoding='unicode'))
        partial=dict(completed_outputs={'a':'PRIVATE_PARTIAL'},sample_id='b')
        suite=module.task_junit(partial,{'error':'RuntimeError'},2)
        self.assertEqual(suite.attrib['tests'],'2');self.assertEqual(suite.attrib['skipped'],'1')
        self.assertEqual(suite.attrib['errors'],'1');self.assertEqual(suite.attrib['failures'],'0')
        self.assertEqual(suite.findall('testcase')[1].get('name'),'execution.b')
        self.assertNotIn('PRIVATE_PARTIAL',ET.tostring(suite,encoding='unicode'))
        evaluated['passed']=True
        suite=module.task_junit(dict(evaluation_result=evaluated),{'error':'trace_export_failed','trace_export_failed':True},2)
        self.assertEqual(suite.attrib['tests'],'4');self.assertEqual(suite.attrib['errors'],'1')
        self.assertEqual(suite.attrib['failures'],'0')
        suite=module.task_junit({}, {'error':'invalid_request'},2)
        self.assertEqual(suite.attrib['tests'],'1');self.assertEqual(suite.attrib['errors'],'1')

    def test_task_output_byte_bounds_and_score_delivery_recovery(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):self.scored=[];self.failure=None
            def metrics(self):return dict(max_metrics_per_run=6,metrics=[dict(id='exact_match',reference_requirement='required')])
            def request(self,path):return dict(id='data',project_id='default',version=1,sha256='hash',samples=[dict(id='a',input='one',expected_output='ok')])
            def evaluate_outputs(self,*args,**kwargs):
                self.scored.append(args)
                if self.failure:raise self.failure
                return {'passed':True}
        client=Fake()
        for answer in ('x'*64000,'\u0430'*32000):
            self.assertTrue(client.evaluate_task('data',1,lambda _:answer,['exact_match'])['passed'])
        before=len(client.scored)
        for answer in ('x'*64001,'\u0430'*32001):
            with self.assertRaises(ValueError) as raised:client.evaluate_task('data',1,lambda _:answer,['exact_match'])
            self.assertEqual(raised.exception.completed_outputs,{})
        self.assertEqual(len(client.scored),before)
        original=EvaluationError('response_lost');client.failure=original;calls=[]
        with self.assertRaises(EvaluationError) as raised:client.evaluate_task('data',1,lambda text:calls.append(text) or 'saved-answer',['exact_match'])
        self.assertIs(raised.exception,original);self.assertEqual(calls,['one'])
        self.assertEqual(original.completed_outputs,{'a':'saved-answer'})
        self.assertEqual(original.evaluation_phase,'score_delivery');self.assertEqual(original.dataset_version,1)
        async def check():
            calls.clear()
            async def task(text):calls.append(text);return 'async-answer'
            with self.assertRaises(EvaluationError) as raised:await client.evaluate_task_async('data',1,task,['exact_match'])
            self.assertIs(raised.exception,original);self.assertEqual(calls,['one'])
            self.assertEqual(original.completed_outputs,{'a':'async-answer'})
            self.assertEqual(original.started_sample_ids,['a'])
            self.assertEqual(original.evaluation_phase,'score_delivery')
            client.failure=None
            async def oversize(text):return '\u0430'*32001
            before=len(client.scored)
            with self.assertRaises(ValueError):await client.evaluate_task_async('data',1,oversize,['exact_match'])
            self.assertEqual(len(client.scored),before)
        asyncio.run(check())

    def test_task_evaluation_preflight_partial_failure_and_once_only(self):
        class Fake(Studio):
            def __init__(self):self.scored=[];self.samples=[dict(id='a',input='one',expected_output='ok'),dict(id='b',input='two',expected_output='ok')]
            def metrics(self):return dict(max_metrics_per_run=6,metrics=[dict(id='exact_match',reference_requirement='required')])
            def request(self,path):return dict(id='data',project_id='default',version=1,sha256='hash',samples=self.samples)
            def evaluate_outputs(self,*args,**kwargs):self.scored.append((args,kwargs));return {'passed':True}
        client=Fake();called=[]
        result=client.evaluate_task('data',1,lambda text:called.append(text) or 'ok',['exact_match'])
        self.assertTrue(result['passed']);self.assertEqual(called,['one','two'])
        self.assertEqual(client.scored[0][0][2],{'a':'ok','b':'ok'})
        client.scored.clear();called.clear()
        original=RuntimeError('task-failed')
        def broken(text):
            called.append(text)
            if text=='two':raise original
            return 'first-output'
        with self.assertRaises(RuntimeError) as raised:client.evaluate_task('data',1,broken,['exact_match'])
        self.assertIs(raised.exception,original)
        self.assertEqual(original.completed_outputs,{'a':'first-output'})
        self.assertEqual(original.sample_id,'b');self.assertEqual(client.scored,[])
        called.clear();client.samples[1]['expected_output']=None
        with self.assertRaises(ValueError):client.evaluate_task('data',1,broken,['exact_match'])
        self.assertEqual(called,[])
        with self.assertRaises(ValueError):client.evaluate_task('data',1,broken,['unknown'])
        with self.assertRaises(ValueError):client.evaluate_task('data',1,broken,['exact_match'],min_scores={'exact_match':2})
        async def asynchronous(text):return text
        with self.assertRaises(ValueError):client.evaluate_task('data',1,asynchronous,['exact_match'])
        client.samples[1]['expected_output']='ok'
        client.scored_outputs=lambda _:dict(dataset_id='other',dataset_version=1,dataset_sha256='hash',metrics=['exact_match'])
        with self.assertRaises(EvaluationError):client.evaluate_task('data',1,broken,['exact_match'],baseline_id='baseline')
        self.assertEqual(called,[])

    def test_wait_network_timeout_at_deadline_keeps_timeout_reason(self):
        import allpaka_studio
        original=EvaluationError('connection_error')
        class Fake(Studio):
            def run(self,*args,**kwargs):raise original
            def judge_run(self,*args,**kwargs):raise original
        client=Fake()
        for wait,reason in ((client.wait,'evaluation_timeout'),(client.wait_judges,'judge_timeout')):
            with patch.object(allpaka_studio.time,'monotonic',side_effect=[0,0,1.1]):
                with self.assertRaises(EvaluationError) as raised:wait('run',timeout=1)
            self.assertEqual(raised.exception.reason,reason);self.assertEqual(raised.exception.run_id,'run')
            with patch.object(allpaka_studio.time,'monotonic',side_effect=[0,0,0.1]):
                with self.assertRaises(EvaluationError) as raised:wait('run',timeout=1)
            self.assertIs(raised.exception,original)

    def test_trace_export_cli_new_file_and_racing_writer(self):
        import importlib.util
        import tempfile
        import json
        import io
        from contextlib import redirect_stdout
        spec=importlib.util.spec_from_file_location('trace_export_cli',pathlib.Path(__file__).with_name('export-studio-traces.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        packet=dict(kind='trace_export_batch',schema_version=1,project_id='default',trace_count=1,provider_calls=0,privacy={'feedback_included':False},traces=[{'trace':{'id':'trace','project_id':'default'},'feedback':None}])
        with tempfile.TemporaryDirectory() as directory:
            target=pathlib.Path(directory)/'export.json';calls=[]
            class Fake:
                def __init__(self,*args):pass
                def export_traces(self,*args,**kwargs):calls.append((args,kwargs));return packet
            def invoke():
                with patch.object(module,'Studio',Fake),patch.object(sys,'argv',['export-studio-traces.py','--base-url','http://127.0.0.1:8100','--trace-id','trace','--output',str(target)]),redirect_stdout(io.StringIO()):return module.main()
            self.assertEqual(invoke(),0);self.assertEqual(json.loads(target.read_text()),packet)
            self.assertEqual(invoke(),2);self.assertEqual(len(calls),1)
            target.unlink()
            def racing(self,*args,**kwargs):target.write_text('other-writer');return packet
            with patch.object(Fake,'export_traces',racing):self.assertEqual(invoke(),2)
            self.assertEqual(target.read_text(),'other-writer')
            self.assertEqual(list(pathlib.Path(directory).glob('.trace-export-*')),[])
            target.unlink();packet['traces'][0]['trace']['project_id']='foreign'
            self.assertEqual(invoke(),2);self.assertFalse(target.exists())

    def test_trace_export_cli_jsonl_preserves_packets_and_unicode(self):
        import importlib.util
        import tempfile
        import json
        import io
        from contextlib import redirect_stdout
        spec=importlib.util.spec_from_file_location('trace_export_jsonl',pathlib.Path(__file__).with_name('export-studio-traces.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        items=[dict(kind='trace_export',schema_version=1,trace=dict(id=i,project_id='default'),feedback={'comment':'\u043e\u0442\u0437\u044b\u0432\nsecond line'},privacy={'feedback_included':True},provider_calls=0) for i in ('second','first')]
        packet=dict(kind='trace_export_batch',schema_version=1,project_id='default',trace_count=2,provider_calls=0,privacy={'feedback_included':True},traces=items)
        class Fake:
            def __init__(self,*args):pass
            def export_traces(self,*args,**kwargs):return packet
        with tempfile.TemporaryDirectory() as directory:
            target=pathlib.Path(directory)/'export.jsonl'
            output=io.StringIO()
            argv=['export-studio-traces.py','--base-url','http://127.0.0.1:8100','--trace-id','second','--trace-id','first','--include-feedback','--format','jsonl','--output',str(target)]
            with patch.object(module,'Studio',Fake),patch.object(sys,'argv',argv),redirect_stdout(output):self.assertEqual(module.main(),0)
            lines=target.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines),2);self.assertEqual([json.loads(line) for line in lines],items)
            summary=json.loads(output.getvalue());self.assertEqual(summary['format'],'jsonl');self.assertEqual(summary['trace_count'],2)
            self.assertNotIn('second line',output.getvalue())
            target.unlink();packet['traces'].reverse()
            with patch.object(module,'Studio',Fake),patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):self.assertEqual(module.main(),2)
            self.assertFalse(target.exists());self.assertEqual(list(pathlib.Path(directory).glob('.trace-export-*')),[])

    def test_bulk_trace_export_bounds_explicit_feedback_and_order(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body,**kwargs):self.calls.append((path,body,kwargs));return body
        client=Fake()
        packet=client.export_traces('default',['second','first'])
        self.assertEqual(packet,dict(project_id='default',trace_ids=['second','first'],include_feedback=False))
        self.assertEqual(client.calls[0][0],'observability/trace-exports')
        self.assertTrue(client.export_traces('default',['first'],include_feedback=True)['include_feedback'])
        before=len(client.calls)
        for ids in ([],['same','same'],['x']*101,'first',[None]):
            with self.assertRaises(ValueError):client.export_traces('default',ids)
        with self.assertRaises(ValueError):client.export_traces('default',['first'],include_feedback=1)
        self.assertEqual(len(client.calls),before)

    def test_export_trace_requires_explicit_feedback_selection(self):
        class Fake(Studio):
            def request(self,path,body=None,**kwargs):return path
        self.assertEqual(Fake().export_trace('trace-id'),'observability/traces/trace-id/export?include_feedback=false')
        self.assertEqual(Fake().export_trace('trace-id',include_feedback=True),'observability/traces/trace-id/export?include_feedback=true')
        with self.assertRaises(ValueError):Fake().export_trace('trace-id',include_feedback=1)
    def test_trace_ingestion_key_is_explicit_without_automatic_retry(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,**kwargs):self.calls.append(body);return body
        client=Fake();client.ingest_trace('default','request',100,[],idempotency_key='receipt-key')
        self.assertEqual(client.calls[0]['idempotency_key'],'receipt-key')
        client.ingest_trace('default','request',100,[])
        self.assertNotIn('idempotency_key',client.calls[1])
        self.assertEqual(len(client.calls),2)
    def test_anthropic_stream_cumulative_usage_terminal_and_async(self):
        import asyncio
        import json
        class Fake(Studio):
            def __init__(self):self.delivered=[]
            def ingest_trace(self,*args,**kwargs):self.delivered.append(args);return {'id':'trace'}
        class Stream:
            def __init__(self,events):self.events=iter(events);self.closed=0
            def __iter__(self):return self
            def __next__(self):return next(self.events)
            def close(self):self.closed+=1
        start={'type':'message_start','message':{'type':'message','usage':{'input_tokens':10,'output_tokens':1,'cache_creation_input_tokens':3,'cache_read_input_tokens':5},'content':'PRIVATE_START'}}
        events=[start,{'type':'content_block_delta','delta':'PRIVATE_DELTA'},
            {'type':'message_delta','usage':{'output_tokens':4}},
            {'type':'message_delta','usage':{'output_tokens':7,'input_tokens':12}},{'type':'message_stop'}]
        client=Fake();raw=Stream(events)
        wrapped=client.track_anthropic_messages_stream(lambda **kwargs:raw,provider_id='anthropic',model_id='model')
        with client.trace('default','anthropic-stream') as trace:
            with wrapped(model='model',stream=True) as stream:
                received=list(stream)
        self.assertIs(received[0],start);self.assertEqual(raw.closed,1)
        self.assertEqual(trace.spans[1]['status'],'completed')
        self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=20,output_tokens=7,cache_read_input_tokens=5,cache_creation_input_tokens=3))
        for stream_events,outcome in [([start],'interrupted'),([{'type':'message_stop'}],'interrupted'),([{'type':'message_start'} ,{'type':'message_stop'}],'interrupted'),([{'type':'error','error':'PRIVATE_ERROR'}],'failed'),([start,start,{'type':'message_stop'}],'failed')]:
            raw=Stream(stream_events)
            with client.trace('default','stream-outcome') as trace:list(wrapped(model='model',stream=True))
            self.assertEqual(trace.spans[1]['status'],outcome)
        async def check():
            class AsyncStream:
                def __init__(self):self.events=iter(events);self.closed=0
                def __aiter__(self):return self
                async def __anext__(self):
                    try:return next(self.events)
                    except StopIteration:raise StopAsyncIteration
                async def close(self):self.closed+=1
            raw=AsyncStream()
            async def create(**kwargs):return raw
            wrapped=client.track_anthropic_messages_stream(create,provider_id='anthropic',model_id='model')
            with client.trace('default','async-anthropic-stream') as trace:
                async with await wrapped(model='model',stream=True) as stream:self.assertEqual(len([event async for event in stream]),5)
            self.assertEqual(raw.closed,1);self.assertEqual(trace.spans[1]['status'],'completed')
            self.assertEqual(trace.spans[1]['usage']['output_tokens'],7)
        asyncio.run(check())
        self.assertNotIn('PRIVATE_',json.dumps(client.delivered))

    def test_anthropic_messages_adapter_input_total_async_and_privacy(self):
        import asyncio
        import json
        from types import SimpleNamespace
        class Fake(Studio):
            def __init__(self):self.delivered=[]
            def ingest_trace(self,*args,**kwargs):self.delivered.append(args);return {'id':'trace'}
        client=Fake();calls=[]
        response=SimpleNamespace(usage=SimpleNamespace(input_tokens=10,output_tokens=5,cache_creation_input_tokens=3,cache_read_input_tokens=7),content='PRIVATE_CONTENT',id='PRIVATE_ID')
        def create(**kwargs):calls.append(kwargs);return response
        wrapped=client.track_anthropic_messages(create,provider_id='anthropic',model_id='claude-test')
        self.assertIs(wrapped(model='claude-test',messages='PRIVATE_INPUT'),response)
        self.assertEqual(client.delivered,[])
        with client.trace('default','anthropic') as trace:
            self.assertIs(wrapped(model='claude-test',messages='PRIVATE_INPUT',system='PRIVATE_SYSTEM'),response)
        self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=20,output_tokens=5,cache_read_input_tokens=7,cache_creation_input_tokens=3))
        before=len(calls)
        with self.assertRaises(ValueError):wrapped(model='claude-test',stream=True)
        with self.assertRaises(ValueError):wrapped(model='different')
        self.assertEqual(len(calls),before)
        for cache in (None,True,-1,2**64):
            response.usage.cache_creation_input_tokens=cache
            with client.trace('default','unknown-input') as trace:wrapped(model='claude-test')
            self.assertEqual(trace.spans[1]['usage'],{'output_tokens':5})
        response.usage.input_tokens=2**64-1;response.usage.cache_creation_input_tokens=1
        with client.trace('default','overflow-input') as trace:wrapped(model='claude-test')
        self.assertEqual(trace.spans[1]['usage'],{'output_tokens':5})
        async def check():
            result={'usage':{'input_tokens':4,'output_tokens':2,'cache_creation_input_tokens':0,'cache_read_input_tokens':0},'content':'PRIVATE_ASYNC'}
            async def create(**kwargs):return result
            wrapped=client.track_anthropic_messages(create,provider_id='anthropic',model_id='claude-test')
            with client.trace('default','async-anthropic') as trace:self.assertIs(await wrapped(model='claude-test'),result)
            self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=4,output_tokens=2,cache_read_input_tokens=0,cache_creation_input_tokens=0))
            original=RuntimeError('PRIVATE_ERROR')
            async def broken(**kwargs):raise original
            wrapped=client.track_anthropic_messages(broken,provider_id='anthropic',model_id='claude-test')
            with self.assertRaises(RuntimeError) as raised:
                with client.trace('default','failed-anthropic') as trace:await wrapped(model='claude-test')
            self.assertIs(raised.exception,original);self.assertEqual(trace.spans[1]['status'],'failed')
        asyncio.run(check())
        self.assertNotIn('PRIVATE_',json.dumps(client.delivered))

    def test_responses_stream_terminal_outcomes_and_private_payloads(self):
        import asyncio
        import json
        class Fake(Studio):
            def __init__(self):self.delivered=[]
            def ingest_trace(self,*args,**kwargs):self.delivered.append(args);return {'id':'trace'}
        class Stream:
            def __init__(self,events):self.events=iter(events);self.closed=0
            def __iter__(self):return self
            def __next__(self):return next(self.events)
            def close(self):self.closed+=1
        client=Fake();raw=None
        def create(**kwargs):return raw
        wrapped=client.track_openai_responses_stream(create,provider_id='local',model_id='model')
        for event_type,status,outcome in [('response.completed','completed','completed'),('response.failed','failed','failed'),('response.incomplete','incomplete','failed')]:
            event={'type':event_type,'response':{'status':status,'usage':{'input_tokens':7,'output_tokens':2,'input_tokens_details':{'cached_tokens':3}},'output':'PRIVATE_OUTPUT','error':'PRIVATE_ERROR'}}
            delta={'type':'response.output_text.delta','delta':'PRIVATE_DELTA'}
            raw=Stream([delta,event])
            with client.trace('default','outcome-'+status) as trace:
                with wrapped(model='model',stream=True) as stream:
                    events=list(stream)
            self.assertIs(events[0],delta);self.assertIs(events[1],event)
            self.assertEqual(trace.spans[1]['status'],outcome)
            self.assertEqual({key:value for key,value in trace.spans[1]['usage'].items() if key!='first_text_ms'},dict(input_tokens=7,output_tokens=2,cache_read_input_tokens=3))
            self.assertIs(type(trace.spans[1]['usage']['first_text_ms']),int)
            self.assertEqual(raw.closed,1)
        raw=Stream([{'type':'response.failed','response':{'status':'failed'}},
                    {'type':'response.completed','response':{'status':'completed'}}])
        with client.trace('default','sticky-failure') as trace:list(wrapped(model='model',stream=True))
        self.assertEqual(trace.spans[1]['status'],'failed')
        for events in ([],[{'type':'response.completed','response':{'status':'failed'}}],[{'type':[]}],[{'type':'response.created'}]):
            raw=Stream(events)
            with client.trace('default','missing-terminal') as trace:list(wrapped(model='model',stream=True))
            self.assertEqual(trace.spans[1]['status'],'interrupted')
            self.assertEqual(trace.spans[1]['usage'],{})
        async def check():
            class AsyncStream:
                def __init__(self):self.events=iter([{'type':'response.completed','response':{'status':'completed','usage':{'input_tokens':4,'output_tokens':1},'output':'PRIVATE_ASYNC'}}]);self.closed=0
                def __aiter__(self):return self
                async def __anext__(self):
                    try:return next(self.events)
                    except StopIteration:raise StopAsyncIteration
                async def close(self):self.closed+=1
            raw=AsyncStream()
            async def create(**kwargs):return raw
            wrapped=client.track_openai_responses_stream(create,provider_id='local',model_id='model')
            with client.trace('default','async-responses-stream') as trace:
                async with await wrapped(model='model',stream=True) as stream:
                    self.assertEqual(len([event async for event in stream]),1)
            self.assertEqual(raw.closed,1);self.assertEqual(trace.spans[1]['status'],'completed')
            self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=4,output_tokens=1))
            with self.assertRaises(ValueError):await wrapped(model='model',stream=True,background=True)
        asyncio.run(check())
        self.assertNotIn('PRIVATE_',json.dumps(client.delivered))

    def test_async_openai_stream_exhaustion_cancellation_and_reader_scope(self):
        import asyncio
        import json
        class Fake(Studio):
            def __init__(self):self.delivered=[]
            def ingest_trace(self,*args,**kwargs):self.delivered.append(args);return {'id':'trace'}
        class Stream:
            def __init__(self,chunks=(),error=None,block=None):self.chunks=iter(chunks);self.error=error;self.block=block;self.closed=0
            def __aiter__(self):return self
            async def __anext__(self):
                if self.block:
                    self.block.set();await asyncio.Event().wait()
                try:return next(self.chunks)
                except StopIteration:
                    if self.error:raise self.error
                    raise StopAsyncIteration
            async def close(self):self.closed+=1
        async def next_chunk(stream):return await stream.__anext__()
        async def check():
            client=Fake();chunks=[{'choices':'PRIVATE_ASYNC_CHUNK'},{'usage':{'prompt_tokens':8,'completion_tokens':2}}]
            raw=Stream(chunks)
            async def create(**kwargs):return raw
            wrapped=client.track_openai_chat_stream(create,provider_id='local',model_id='model')
            with client.trace('default','async-stream') as trace:
                async with await wrapped(model='model',stream=True) as stream:
                    self.assertIs(await next_chunk(stream),chunks[0])
                    with trace.span('sibling'):pass
                    remaining=[chunk async for chunk in stream]
                    self.assertIs(remaining[0],chunks[1])
            self.assertEqual(raw.closed,1);self.assertEqual(trace.spans[1]['status'],'completed')
            self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=8,output_tokens=2))
            self.assertEqual(trace.spans[2]['parent_id'],0)
            original=RuntimeError('PRIVATE_ERROR');raw=Stream(error=original)
            with self.assertRaises(RuntimeError) as raised:
                with client.trace('default','failed') as trace:
                    stream=await wrapped(model='model',stream=True);await next_chunk(stream)
            self.assertIs(raised.exception,original);self.assertEqual(raw.closed,1)
            self.assertEqual(trace.spans[1]['status'],'failed')
            entered=asyncio.Event();raw=Stream(block=entered)
            with client.trace('default','cancelled') as trace:
                stream=await wrapped(model='model',stream=True)
                read=asyncio.create_task(next_chunk(stream));await entered.wait()
                with self.assertRaisesRegex(ValueError,'one active reader'):await next_chunk(stream)
                read.cancel()
                with self.assertRaises(asyncio.CancelledError):await read
                self.assertEqual(trace._parent.get(),0)
                await stream.aclose()
            self.assertEqual(raw.closed,1);self.assertEqual(trace.spans[1]['status'],'interrupted')
            raw=Stream(chunks)
            with client.trace('default','early') as trace:
                async with await wrapped(model='model',stream=True) as stream:await next_chunk(stream)
            self.assertEqual(raw.closed,1);self.assertEqual(trace.spans[1]['status'],'interrupted')
            cancellation=asyncio.CancelledError()
            async def cancelled_create(**kwargs):raise cancellation
            cancelled=client.track_openai_chat_stream(cancelled_create,provider_id='local',model_id='model')
            with self.assertRaises(asyncio.CancelledError) as raised:
                with client.trace('default','creation-cancelled') as trace:await cancelled(model='model',stream=True)
            self.assertIs(raised.exception,cancellation);self.assertEqual(trace.spans[1]['status'],'interrupted')
            class CloseFailure(Stream):
                async def close(self):raise RuntimeError('PRIVATE_CLOSE_ERROR')
            raw=CloseFailure(chunks)
            with self.assertRaises(RuntimeError) as raised:
                with client.trace('default','consumer-error') as trace:
                    async with await wrapped(model='model',stream=True) as stream:
                        await next_chunk(stream);raise original
            self.assertIs(raised.exception,original);self.assertEqual(trace.spans[1]['status'],'interrupted')
            self.assertNotIn('PRIVATE_',json.dumps(client.delivered))
        asyncio.run(check())

    def test_openai_stream_lifecycle_chunks_usage_and_cleanup(self):
        import json
        class Fake(Studio):
            def __init__(self):self.delivered=[]
            def ingest_trace(self,*args,**kwargs):self.delivered.append(args);return {'id':'trace'}
        class Stream:
            def __init__(self,chunks,error=None):self.chunks=iter(chunks);self.closed=0;self.error=error
            def __iter__(self):return self
            def __next__(self):
                try:return next(self.chunks)
                except StopIteration:
                    if self.error:raise self.error
                    raise
            def close(self):self.closed+=1
        client=Fake();chunks=[{'choices':'PRIVATE_CHUNK'},{'usage':{'prompt_tokens':7,'completion_tokens':3}}]
        streams=[]
        def create(**kwargs):
            stream=Stream(chunks);streams.append(stream);return stream
        wrapped=client.track_openai_chat_stream(create,provider_id='local',model_id='model')
        with client.trace('default','stream') as trace:
            stream=wrapped(model='model',stream=True,messages='PRIVATE_INPUT')
            with trace.span('sibling'):pass
            self.assertIs(next(stream),chunks[0])
            self.assertIs(next(stream),chunks[1])
            self.assertEqual(trace._parent.get(),0)
            with self.assertRaises(StopIteration):next(stream)
        self.assertEqual(streams[0].closed,1)
        self.assertEqual(trace.spans[1]['status'],'completed')
        self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=7,output_tokens=3))
        self.assertEqual(trace.spans[2]['parent_id'],0)
        self.assertNotIn('PRIVATE_',json.dumps(client.delivered))
        with client.trace('default','early') as trace:
            with wrapped(model='model',stream=True) as stream:next(stream)
            stream.close()
        self.assertEqual(trace.spans[1]['status'],'interrupted');self.assertEqual(streams[-1].closed,1)
        original=RuntimeError('PRIVATE_STREAM_ERROR')
        broken_stream=Stream([],original)
        broken=client.track_openai_chat_stream(lambda **kwargs:broken_stream,provider_id='local',model_id='model')
        with self.assertRaises(RuntimeError) as raised:
            with client.trace('default','broken') as trace:list(broken(model='model',stream=True))
        self.assertIs(raised.exception,original);self.assertEqual(trace.spans[1]['status'],'failed')
        self.assertEqual(broken_stream.closed,1)
        with self.assertRaisesRegex(ValueError,'active children'):
            with client.trace('default','unfinished') as trace:stream=wrapped(model='model',stream=True)
        self.assertEqual(trace.spans[1]['status'],'interrupted')
        stream.close();self.assertEqual(streams[-1].closed,1)
        before=len(streams)
        with self.assertRaises(ValueError):wrapped(model='model')
        self.assertEqual(len(streams),before)

    def test_openai_responses_adapter_sync_async_scope_and_usage(self):
        import asyncio
        import json
        from types import SimpleNamespace
        class Fake(Studio):
            def __init__(self):self.delivered=[]
            def ingest_trace(self,*args,**kwargs):self.delivered.append(args);return {'id':'trace'}
        client=Fake();calls=[]
        response=SimpleNamespace(usage=SimpleNamespace(input_tokens=10,output_tokens=6,input_tokens_details=SimpleNamespace(cached_tokens=3)),output='PRIVATE_OUTPUT',id='PRIVATE_ID')
        def create(**kwargs):calls.append(kwargs);return response
        wrapped=client.track_openai_responses(create,provider_id='local',model_id='model')
        self.assertIs(wrapped(model='model',input='PRIVATE_INPUT'),response)
        self.assertEqual(client.delivered,[])
        with client.trace('default','responses') as trace:
            self.assertIs(wrapped(model='model',input='PRIVATE_INPUT',instructions='PRIVATE_INSTRUCTIONS'),response)
        self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=10,output_tokens=6,cache_read_input_tokens=3))
        self.assertNotIn('PRIVATE_',json.dumps(client.delivered))
        before=len(calls)
        for kwargs in (dict(model='model',stream=True),dict(model='model',background=True),dict(model='other')):
            with self.assertRaises(ValueError):wrapped(**kwargs)
        self.assertEqual(len(calls),before)
        other=Fake()
        with other.trace('default','foreign') as foreign:wrapped(model='model')
        self.assertEqual(len(foreign.spans),1)
        async def check():
            original=RuntimeError('PRIVATE_EXCEPTION')
            async def broken(**kwargs):raise original
            wrapped=client.track_openai_responses(broken,provider_id='local',model_id='model')
            with self.assertRaises(RuntimeError) as raised:
                with client.trace('default','failure') as failed:await wrapped(model='model')
            self.assertIs(raised.exception,original);self.assertEqual(failed.spans[1]['status'],'failed')
            class AsyncCreate:
                async def __call__(self,**kwargs):return {'usage':{'input_tokens':5,'output_tokens':0,'input_tokens_details':{'cached_tokens':6}},'output':'PRIVATE_ASYNC'}
            wrapped=client.track_openai_responses(AsyncCreate(),provider_id='local',model_id='model')
            with client.trace('default','async-responses') as trace:
                result=await wrapped(model='model')
            self.assertEqual(result['output'],'PRIVATE_ASYNC')
            self.assertEqual(trace.spans[1]['usage'],dict(input_tokens=5,output_tokens=0))
        asyncio.run(check())
        self.assertNotIn('PRIVATE_',json.dumps(client.delivered))

    def test_openai_chat_adapter_metadata_usage_and_original_errors(self):
        import asyncio
        import json
        from types import SimpleNamespace
        class Fake(Studio):
            def __init__(self):self.delivered=[]
            def ingest_trace(self,*args,**kwargs):self.delivered.append(args);return {'id':'trace'}
        client=Fake();calls=[]
        response=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12,completion_tokens=4,prompt_tokens_details=SimpleNamespace(cached_tokens=8)),choices=['PRIVATE_ANSWER'],id='PRIVATE_RESPONSE')
        def create(**kwargs):calls.append(kwargs);return response
        wrapped=client.track_openai_chat(create,provider_id='local',model_id='model/v1')
        self.assertIs(wrapped(model='model/v1',messages=['PRIVATE_PROMPT']),response)
        self.assertEqual(client.delivered,[])
        with client.trace('default','completion') as trace:
            self.assertIs(wrapped(model='model/v1',messages=['PRIVATE_PROMPT'],api_key='PRIVATE_KEY'),response)
        span=trace.spans[1]
        self.assertEqual(span['kind'],'model');self.assertEqual(span['name'],'model/v1')
        self.assertEqual(span['provider_id'],'local')
        self.assertEqual(span['usage'],dict(input_tokens=12,output_tokens=4,cache_read_input_tokens=8))
        self.assertEqual(span['status'],'completed')
        self.assertNotIn('PRIVATE_',json.dumps(client.delivered))
        before=len(calls)
        for kwargs in (dict(model='model/v1',stream=True),dict(model='different')):
            with self.assertRaises(ValueError):wrapped(**kwargs)
        self.assertEqual(len(calls),before)
        original=RuntimeError('PRIVATE_ERROR')
        def broken(**kwargs):raise original
        broken=client.track_openai_chat(broken,provider_id='local',model_id='model')
        with self.assertRaises(RuntimeError) as raised:
            with client.trace('default','broken') as failed:broken(model='model')
        self.assertIs(raised.exception,original);self.assertEqual(failed.spans[1]['status'],'failed')
        self.assertNotIn('PRIVATE_ERROR',json.dumps(client.delivered))
        async def check():
            async def create(**kwargs):return {'usage':{'prompt_tokens':True,'completion_tokens':3,'prompt_tokens_details':{'cached_tokens':99}},'choices':'PRIVATE_ASYNC'}
            wrapped=client.track_openai_chat(create,provider_id='local',model_id='async-model')
            with client.trace('default','async') as trace:
                answer=await wrapped(model='async-model')
            self.assertEqual(answer['choices'],'PRIVATE_ASYNC')
            self.assertEqual(trace.spans[1]['usage'],{'output_tokens':3})
            cancellation=asyncio.CancelledError()
            async def cancelled(**kwargs):raise cancellation
            wrapped=client.track_openai_chat(cancelled,provider_id='local',model_id='async-model')
            with self.assertRaises(asyncio.CancelledError) as raised:
                with client.trace('default','cancelled') as trace:await wrapped(model='async-model')
            self.assertIs(raised.exception,cancellation);self.assertEqual(trace.spans[1]['status'],'interrupted')
        asyncio.run(check())

    def test_tracking_decorators_preserve_values_scope_and_async_parents(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):self.sent=[]
            def ingest_trace(self,*args):self.sent.append(args);return {'id':'trace-fixture'}
        client=Fake();other=Fake()
        @client.track('inner','model')
        def inner(value):return {'PRIVATE_RESULT':value}
        @client.track('outer','agent')
        def outer(value):return inner(value)
        self.assertEqual(outer('PRIVATE_ARGUMENT'),{'PRIVATE_RESULT':'PRIVATE_ARGUMENT'})
        self.assertEqual(client.sent,[])
        with client.trace('default','sync') as trace:
            self.assertEqual(outer('PRIVATE_ARGUMENT'),{'PRIVATE_RESULT':'PRIVATE_ARGUMENT'})
        self.assertEqual([span['parent_id'] for span in trace.spans],[None,0,1])
        self.assertNotIn('PRIVATE_ARGUMENT',str(client.sent));self.assertNotIn('PRIVATE_RESULT',str(client.sent))
        with other.trace('default','other') as other_trace:outer('value')
        self.assertEqual(len(other_trace.spans),1)
        @client.track('async-model','model')
        async def asynchronous(value):
            await asyncio.sleep(0)
            return value
        async def exercise():
            with client.trace('default','async') as trace:
                self.assertEqual(await asyncio.gather(asynchronous(1),asynchronous(2)),[1,2])
            return trace
        asynchronous_trace=asyncio.run(exercise())
        self.assertEqual([span['parent_id'] for span in asynchronous_trace.spans],[None,0,0])
        def generator():yield 1
        with self.assertRaises(ValueError):client.track('generator')(generator)

    def test_trace_context_metadata_exception_and_export_failure(self):
        class Fake(Studio):
            def __init__(self):self.sent=[];self.fail=False
            def ingest_trace(self,*args):
                self.sent.append(args)
                if self.fail:raise EvaluationError('connection_error')
                return {'id':'trace-fixture'}
        client=Fake()
        with client.trace('default','test') as trace:
            with trace.span('model-v1','model',provider_id='local') as span:span.set_usage(input_tokens=3,cost=0.2,cost_currency='USD')
        self.assertEqual(trace.receipt['id'],'trace-fixture')
        self.assertEqual(trace.spans[1]['parent_id'],0)
        self.assertEqual(trace.spans[1]['status'],'completed')
        self.assertEqual(trace.spans[1]['provider_id'],'local')
        self.assertLessEqual(trace.spans[1]['started_ms']+trace.spans[1]['duration_ms'],trace.spans[0]['started_ms']+trace.spans[0]['duration_ms'])
        client.fail=True
        original=RuntimeError('PRIVATE_EXCEPTION_BODY')
        with self.assertRaises(RuntimeError) as raised:
            with client.trace('default','error') as failed:
                with failed.span('tool-v1'):raise original
        self.assertIs(raised.exception,original)
        self.assertEqual(failed.spans[1]['status'],'failed')
        self.assertIsInstance(failed.export_error,EvaluationError)
        self.assertNotIn('PRIVATE_EXCEPTION_BODY',str(client.sent))
        with self.assertRaises(EvaluationError):
            with client.trace('default','export-error'):pass

    def test_trace_explicit_retry_uses_frozen_keyed_snapshot(self):
        import json
        class Fake(Studio):
            def __init__(self):self.sent=[]
            def ingest_trace(self,*args,**kwargs):
                self.sent.append(json.loads(json.dumps([args,kwargs])))
                if len(self.sent)==1:raise EvaluationError('connection_error')
                return {'id':'trace-keyed','deduplicated':True}
        client=Fake()
        executions=[]
        with self.assertRaises(EvaluationError):
            with client.trace('default','retry',idempotency_key='execution-1') as trace:
                executions.append(1)
                with self.assertRaises(ValueError):trace.export()
                with trace.span('model','model'):pass
        trace.spans[1]['name']='changed-after-close'
        receipt=trace.export()
        self.assertEqual(client.sent[0],client.sent[1])
        self.assertEqual(client.sent[1][1],{'idempotency_key':'execution-1'})
        self.assertIs(trace.export(),receipt)
        self.assertEqual(len(client.sent),2)
        self.assertEqual(executions,[1])
        self.assertIsNone(trace.export_error)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            receipts=list(pool.map(lambda _:trace.export(),range(16)))
        self.assertTrue(all(item is receipt for item in receipts))
        self.assertEqual(len(client.sent),2)
        client=Fake()
        with self.assertRaises(EvaluationError):
            with client.trace('default','unkeyed') as unkeyed:pass
        with self.assertRaises(ValueError):unkeyed.export()
        self.assertEqual(len(client.sent),1)
        with self.assertRaises(ValueError):client.trace('default','bad',idempotency_key='private key')

    def test_trace_async_siblings_have_same_parent_and_no_payload(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):pass
            def ingest_trace(self,*args):return {'id':'trace-fixture'}
        async def exercise():
            with Fake().trace('default','async') as trace:
                async def child(name):
                    with trace.span(name,'model') as span:
                        await asyncio.sleep(0)
                        span.set_usage(output_tokens=2)
                await asyncio.gather(child('a'),child('b'))
            return trace
        trace=asyncio.run(exercise())
        self.assertEqual([span['parent_id'] for span in trace.spans],[None,0,0])
        with self.assertRaises(ValueError):trace.span('PRIVATE TEXT')
        with self.assertRaises(ValueError):
            with Fake().trace('default','bad') as trace:
                with trace.span('a') as span:span.set_usage(api_key='SECRET')

    def test_trace_unfinished_child_and_cancellation_keep_interrupted_evidence(self):
        import asyncio
        class Fake(Studio):
            def __init__(self):pass
            def ingest_trace(self,*args):return {'id':'trace-fixture'}
        with self.assertRaises(ValueError):
            with Fake().trace('default','unfinished') as trace:
                child=trace.span('child');child.__enter__()
        self.assertEqual(trace.receipt['id'],'trace-fixture')
        self.assertEqual([span['status'] for span in trace.spans],['interrupted','interrupted'])
        child.__exit__(None,None,None)
        with self.assertRaises(asyncio.CancelledError):
            with Fake().trace('default','cancel') as cancelled:
                with cancelled.span('child'):raise asyncio.CancelledError()
        self.assertEqual([span['status'] for span in cancelled.spans],['interrupted','interrupted'])

    def test_matrix_preflight_and_partial_failure_evidence(self):
        import copy
        request=dict(dataset_id='data',dataset_version=1,metrics=['exact_match'],
                     settings=dict(project_id='default',mode='chat',allow_writes=False),prompt_template='{{input}}')
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def evaluate(self,request,**kwargs):
                self.calls.append((request,kwargs))
                if len(self.calls)==2:raise EvaluationError('connection_error','second')
                return dict(run=dict(id='first',status='completed',strict_quality=True),passed=True)
        client=Fake();bad=copy.deepcopy(request);bad['dataset_version']=2
        with self.assertRaises(ValueError):client.evaluate_matrix([request,bad])
        self.assertEqual(client.calls,[])
        with self.assertRaises(EvaluationError) as raised:
            client.evaluate_matrix([request,request],labels=['baseline','candidate'])
        error=raised.exception
        self.assertEqual(error.variant_index,1)
        self.assertEqual(error.run_id,'second')
        self.assertEqual(error.matrix_results[0]['label'],'baseline')
        self.assertEqual(client.calls[1][1]['baseline_id'],'first')
        self.assertIsNot(client.calls[0][0],request)

    def test_matrix_persistence_preserves_client_gates_and_partial_evidence(self):
        request=dict(dataset_id='data',dataset_version=1,metrics=['exact_match'],settings=dict(project_id='default',mode='chat',allow_writes=False),prompt_template='{{input}}')
        class Fake(Studio):
            def __init__(self):self.runs=0;self.saves=0;self.fail=False
            def evaluate(self,request,**kwargs):
                self.runs+=1;return dict(run=dict(id='run'+str(self.runs),status='completed',strict_quality=True),passed=False)
            def save_experiment_matrix(self,project,baseline,variants):
                self.saves+=1
                if self.fail:raise EvaluationError('http_500')
                return dict(kind='experiment_matrix',schema_version=1,id='saved',project_id=project,baseline_id=baseline,automatic_promotion=False,passed=True)
        client=Fake();report=client.evaluate_matrix([request,request],persist=True)
        self.assertEqual(client.saves,1);self.assertTrue(report['native_matrix']['passed']);self.assertFalse(report['variants'][1]['result']['passed'])
        broken=Fake();broken.fail=True
        with self.assertRaises(EvaluationError) as caught:broken.evaluate_matrix([request,request],persist=True)
        self.assertTrue(caught.exception.matrix_persistence_failed);self.assertEqual(len(caught.exception.matrix_results),2);self.assertEqual(broken.runs,2);self.assertEqual(broken.saves,1)
        invalid=Fake()
        with self.assertRaises(ValueError):invalid.evaluate_matrix([request,request],persist=1)
        self.assertEqual(invalid.runs,0)

    def test_matrix_retains_paired_gate_results_without_promotion(self):
        request=dict(dataset_id='data',dataset_version=1,metrics=['exact_match'],
                     settings=dict(project_id='default',mode='chat',allow_writes=False),prompt_template='{{input}}')
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def evaluate(self,request,**kwargs):
                self.calls.append(kwargs)
                return dict(run=dict(id='run'+str(len(self.calls)),status='completed',strict_quality=True),
                            passed=len(self.calls)==1,comparison=None if len(self.calls)==1 else dict(regressions=1))
        client=Fake();matrix=client.evaluate_matrix([request,request],require_improvement=True)
        self.assertFalse(matrix['automatic_promotion'])
        self.assertFalse(matrix['variants'][1]['result']['passed'])
        self.assertFalse(client.calls[0]['require_improvement'])
        self.assertTrue(client.calls[1]['require_improvement'])

    def test_provider_doctor_is_explicit_catalog_probe(self):
        class Fake(Studio):
            def request(self,path,body=None,timeout=10):return (path,body,timeout)
        self.assertEqual(Fake().provider_doctor('local','mock'),('providers/local/doctor',{'model':'mock'},15))
        with self.assertRaises(ValueError):Fake().provider_doctor('../foreign')

    def test_judge_preset_is_explicit_and_exclusive(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,**kwargs):self.calls.append((path,body));return body
        client=Fake()
        for preset in ({'id':'source_faithfulness','version':0},{'id':'source_faithfulness','version':True},{'id':'source_faithfulness'}):
            with self.assertRaises(ValueError):client.plan_judges('data',1,{}, {},judge_preset=preset)
        with self.assertRaises(ValueError):client.plan_judges('data',1,{}, {},'Inline',judge_preset={'id':'source_faithfulness','version':1})
        self.assertEqual(client.calls,[])
        body=client.plan_judges('data',1,{}, {},judge_preset={'id':'source_faithfulness','version':1})
        self.assertEqual(body['judge_preset'],{'id':'source_faithfulness','version':1})
        self.assertNotIn('rubric',body)

    def test_rubric_reference_requires_explicit_version_and_single_source(self):
        class Fake(Studio):
            def __init__(self):self.calls=[]
            def request(self,path,body=None,**kwargs):self.calls.append(body);return body
        client=Fake()
        for ref in ({'id':'rubric','version':0},{'id':'rubric','version':True},{'id':'rubric'},{'id':'rubric','version':1,'extra':1}):
            with self.assertRaises(ValueError):client.plan_judges('data',1,{}, {},rubric_ref=ref)
        with self.assertRaises(ValueError):client.plan_judges('data',1,{}, {},'Inline',rubric_ref={'id':'rubric','version':1})
        self.assertEqual(client.calls,[])
        body=client.plan_judges('data',1,{}, {},rubric_ref={'id':'rubric','version':2})
        self.assertEqual(body['rubric_ref'],{'id':'rubric','version':2})
        self.assertNotIn('rubric',body)

    def test_judge_baseline_preflight_does_not_start_incompatible_plan(self):
        class Fake(Studio):
            def __init__(self):self.started=0
            def judge_plan(self,id):return dict(id=id,kind='judge_plan',plan_sha256='hash',rubric=id)
            def judge_run(self,id,**kwargs):return dict(id=id,status='completed',plan_id='old-plan',plan_sha256='hash')
            def start_judges(self,*args):self.started+=1;raise AssertionError('Unexpected model batch')
        client=Fake()
        with self.assertRaisesRegex(EvaluationError,'judge_baseline_mismatch'):
            client.evaluate_judge_plan('new-plan',baseline_id='old-run')
        with self.assertRaises(ValueError):client.evaluate_judge_plan('new-plan',require_improvement=True)
        self.assertEqual(client.started,0)

    def test_experiment_judge_plan_requires_complete_original_outputs(self):
        class Fake(Studio):
            def __init__(self):
                self.calls=[]
                self.result=dict(id='run',status='completed',dataset_id='dataset',dataset_version=2,
                    items=[dict(sample_id='a',status='completed',output='Answer',output_truncated=False)])
            def run(self,*args):return self.result
            def request(self,path,body=None,**kwargs):self.calls.append((path,body));return body
        client=Fake();settings=dict(project_id='project',mode='chat')
        plan=client.plan_experiment_judges('run',settings,'Criterion')
        self.assertEqual(plan['outputs'],{'a':'Answer'})
        self.assertEqual(plan['experiment_id'],'run')
        self.assertEqual(plan['dataset_version'],2)
        for field,value in [('status','failed'),('output_truncated',True),('output',None)]:
            client.result['items'][0][field]=value
            with self.assertRaises(EvaluationError):client.plan_experiment_judges('run',settings,'Criterion')
            client.result['items'][0]=dict(sample_id='a',status='completed',output='Answer',output_truncated=False)
        client.result['items']*=2
        with self.assertRaises(EvaluationError):client.plan_experiment_judges('run',settings,'Criterion')
        self.assertEqual(len(client.calls),1)

    def test_judge_wait_errors_cancel_only_new_batches(self):
        class Fake(Studio):
            def __init__(self):self.cancelled=[]
            def judge_plan(self,*args):return dict(id='plan',kind='judge_plan',plan_sha256='hash')
            def start_judges(self,*args):return dict(id='new-run',plan_id='plan')
            def judge_run(self,*args,**kwargs):raise EvaluationError('network_error')
            def cancel_judges(self,run):self.cancelled.append(run)
        client=Fake()
        with self.assertRaises(EvaluationError):client.wait_judges('existing')
        self.assertEqual(client.cancelled,[])
        with self.assertRaises(EvaluationError) as failure:client.evaluate_judge_plan('plan')
        self.assertEqual(failure.exception.run_id,'new-run')
        self.assertEqual(client.cancelled,['new-run'])
        with self.assertRaises(ValueError):client.evaluate_judge_plan('plan',min_score=float('nan'))
        self.assertEqual(client.cancelled,['new-run'])

    def test_batch_judge_preflight_rejects_bad_outputs_before_provider(self):
        class Fake(Studio):
            def __init__(self):self.judges=0
            def request(self,*args,**kwargs):return dict(id='data',version=1,project_id='project',sha256='hash',samples=[dict(id='a',input='Question',contexts=[],expected_output=None)])
            def judge(self,*args,**kwargs):self.judges+=1;raise AssertionError('Provider should not be called')
        client=Fake();settings=dict(project_id='project',mode='chat',allow_writes=False)
        for outputs in ({},{'other':'Answer'},{'a':1},{'a':'x'*64001}):
            with self.assertRaises(ValueError):client.judge_outputs('data',1,outputs,settings,'Rubric')
        self.assertEqual(client.judges,0)

    def test_judge_receipt_identity_is_checked(self):
        class Fake(Studio):
            def __init__(self,receipt):self.receipt=receipt
            def request(self,*args,**kwargs):return self.receipt
        for receipt in ({'id':'foreign','kind':'llm_judge'},{'id':'judge','kind':'other'},[]):
            with self.assertRaisesRegex(EvaluationError,'invalid_judge_receipt'):Fake(receipt).judgment('judge')

    def test_offline_threshold_validation_does_not_write_receipts(self):
        class Fake(Studio):
            def __init__(self): self.calls=[]
            def request(self,*args,**kwargs): self.calls.append(args);raise AssertionError('Unexpected request')
        client=Fake()
        for threshold in (True,-1,2,float('nan'),float('inf')):
            with self.assertRaises(ValueError):client.evaluate_outputs('dataset',1,{},['exact_match'],min_scores={'exact_match':threshold})
        with self.assertRaises(ValueError):client.evaluate_outputs('dataset',1,{},['exact_match'],require_improvement=True)
        with self.assertRaises(ValueError):client.evaluate_outputs('dataset',1,{},['exact_match'],min_scores={'other':1})
        self.assertEqual(client.calls,[])

    def test_memory_workflow_preserves_reviewed_provenance(self):
        class Fake(Studio):
            def __init__(self): self.calls=[]
            def request(self,path,body=None,timeout=10):
                self.calls.append((path,body,timeout))
                if path == 'memory/proposals/proposal':
                    return dict(id='proposal',project_id='project',notes=[dict(name='Title',content='Fact')])
                return body
        client=Fake()
        saved=client.accept_memory('proposal',0,content='Reviewed fact')
        self.assertEqual(saved['content'],'Reviewed fact')
        self.assertEqual(saved['proposal_source'],dict(proposal_id='proposal',note_index=0))
        self.assertEqual(saved['project_id'],'project')
        self.assertEqual(saved['base_version'],0)
        before=len(client.calls)
        for index in (-1, True, '0'):
            with self.assertRaises(ValueError):client.accept_memory('proposal',index)
        self.assertEqual(len(client.calls),before)
        with self.assertRaises(ValueError):client.accept_memory('proposal',1)
        self.assertEqual(client.calls[-1][0],'memory/proposals/proposal')
        client.extract_memory('session',dict(mode='chat'),2)
        self.assertEqual(client.calls[-1],('sessions/session/memory-proposals',dict(settings=dict(mode='chat'),message_count=2),65))
        client.memory_proposals('session')
        self.assertIsNone(client.calls[-1][1])

    def test_origin_and_artifact_paths_reject_ambiguity(self):
        for url in ('file:///tmp/private', 'http://user:secret@example.com', 'http://example.com/path', 'http://example.com?key=secret', 'http://example.com#fragment'):
            with self.assertRaises(ValueError): Studio(url)
        for value in ('../escape', '', None, 'a/b', 'é', 'a'*81):
            with self.assertRaises(ValueError): Studio._id(value)

    def test_protocol_does_not_follow_redirects_or_echo_error_content(self):
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_): pass
            def do_GET(self):
                calls.append(self.path)
                if self.path=='/api/redirect':
                    self.send_response(302);self.send_header('Location','/api/leaked');self.end_headers()
                elif self.path=='/api/large':
                    self.send_response(200);self.end_headers();self.wfile.write(b' '+b'x'*100)
                elif self.path=='/api/nonfinite':
                    self.send_response(200);self.end_headers();self.wfile.write(b'{"score":NaN}')
                else:
                    self.send_response(400);self.end_headers();self.wfile.write(b'private prompt and credentials')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            studio=Studio(f'http://127.0.0.1:{server.server_port}')
            with self.assertRaisesRegex(EvaluationError,'^http_302$'):studio.request('redirect')
            with self.assertRaisesRegex(EvaluationError,'^http_400$'):studio.request('private')
            self.assertEqual(calls,['/api/redirect','/api/private'])
            with patch('allpaka_studio.MAX_RESPONSE_BYTES',8):
                with self.assertRaisesRegex(EvaluationError,'response_too_large'):studio.request('large')
            with self.assertRaisesRegex(EvaluationError,'invalid_response'):studio.request('nonfinite')
        finally:server.shutdown();server.server_close();thread.join()

    def test_bounds_and_invalid_baseline_do_not_start_a_run(self):
        class Fake(Studio):
            def __init__(self):self.starts=0
            def request(self,path,body=None,timeout=10):
                if body:self.starts+=1
                return dict(id='base',status='failed',strict_quality=False)
        client=Fake()
        for timeout in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):client.evaluate({},timeout=timeout)
        with self.assertRaisesRegex(EvaluationError,'invalid_baseline'):client.evaluate({},baseline_id='base')
        self.assertEqual(client.starts,0)

    def test_wait_receipt_identity_and_status_are_verified(self):
        class Fake(Studio):
            def __init__(self,receipt):self.receipt=receipt
            def run(self,*args,**kwargs):return self.receipt
        for receipt in ({'id':'foreign','status':'completed'}, {'id':'run','status':'invented'}):
            with self.assertRaisesRegex(EvaluationError,'invalid_run_receipt'):Fake(receipt).wait('run')

    def test_keyboard_interrupt_cancels_only_created_run(self):
        class Fake(Studio):
            def __init__(self):self.cancelled=[]
            def request(self,*args,**kwargs):return {'id':'created'}
            def wait(self,*args,**kwargs):raise KeyboardInterrupt()
            def cancel(self,run_id):self.cancelled.append(run_id)
        client=Fake()
        with self.assertRaises(KeyboardInterrupt) as raised:client.evaluate({})
        self.assertEqual(client.cancelled,['created'])
        self.assertEqual(raised.exception.run_id,'created')

    def test_thresholds_fail_closed_and_validate_before_launch(self):
        class Fake(Studio):
            def __init__(self,score):self.score=score;self.starts=0
            def request(self,*args,**kwargs):self.starts+=1;return {"id":"created"}
            def wait(self,*args,**kwargs):return dict(id="created",status="completed",strict_quality=True,mean_scores={"exact_match":self.score})
        request={"metrics":["exact_match"]}
        for score in (None,float('nan'),2):
            with self.assertRaises(EvaluationError):Fake(score).evaluate(request,min_scores={"exact_match":1})
        self.assertFalse(Fake(0.5).evaluate(request,min_scores={"exact_match":1})["passed"])
        self.assertTrue(Fake(1).evaluate(request,min_scores={"exact_match":1})["passed"])
        client=Fake(1)
        for thresholds in ({"exact_match":float('inf')},{"unknown":1},{"exact_match":True}):
            with self.assertRaises(ValueError):client.evaluate(request,min_scores=thresholds)
        self.assertEqual(client.starts,0)

    def test_comparison_connection_error_retains_completed_run_id(self):
        request=dict(dataset_id='data',dataset_version=1,metrics=['json_valid','exact_match'],settings={'project_id':'default'})
        class Fake(Studio):
            def __init__(self):pass
            def run(self,*args,**kwargs):return dict(id='base',status='completed',strict_quality=True,project_id='default',dataset_id='data',dataset_version=1,metrics=['exact_match','json_valid'])
            def request(self,path,body=None,timeout=10):
                if path=='evaluation/compare':raise EvaluationError('connection_error')
                return {'id':'created'}
            def wait(self,*args,**kwargs):return dict(id='created',status='completed',strict_quality=True)
        with self.assertRaises(EvaluationError) as raised:Fake().evaluate(request,baseline_id='base')
        self.assertEqual(raised.exception.run_id,'created')


if __name__=='__main__':unittest.main()
