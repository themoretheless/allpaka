#!/usr/bin/env python3
"""Callback CI command qualification with native-shaped receipts and no network."""
import contextlib,csv,importlib.util,io,json,pathlib,sys,tempfile,unittest,xml.etree.ElementTree as ET
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('evaluation_cli',pathlib.Path(__file__).with_name('evaluate-studio.py'))
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
RealStudio=cli.Studio
class CallbackCI(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.directory.name);self.calls=[]
        self.requirement=dict(evaluator_id='check',evaluator_version=2,metric='quality',min_count=2,min_mean=.75,min_score=.5)
        self.config=self.root/'requirements.json';self.config.write_text(json.dumps([self.requirement]))
        self.packet=dict(kind='callback_evaluation_summary',project_id='default',since_ms=1,until_ms=2,assessment_source='caller_reported',provider_calls=0,automatic_promotion=False,trace_count=3,selected_tasks=3,skipped_tasks=0,completed_assessments=2,failed_assessments=1,metrics=[dict(evaluator_id='check',evaluator_version=2,metric='quality',count=2,mean=.75,min=.5,max=1)])
        owner=self
        class Fake(RealStudio):
            def request(self,path,body=None,timeout=10):owner.calls.append(path);return owner.packet
        self.fake=Fake
    def tearDown(self):self.directory.cleanup()
    def run_cli(self,*extra):
        args=['evaluate-studio.py','--base-url','http://127.0.0.1:1','--callback-requirements-file',str(self.config),'--callback-since-ms','1','--callback-until-ms','2',*extra]
        with patch.object(sys,'argv',args),patch.object(cli,'Studio',self.fake),contextlib.redirect_stdout(io.StringIO()) as output,contextlib.redirect_stderr(io.StringIO()):
            try:code=cli.main()
            except SystemExit as error:code=error.code
        return code,output.getvalue()
    def test_success_reports_and_query(self):
        report=self.root/'report.json';junit=self.root/'report.xml'
        code,_=self.run_cli('--callback-max-failed-assessments','1','--report',str(report),'--junit',str(junit))
        self.assertEqual(code,0);self.assertEqual(len(self.calls),1);self.assertIn('since_ms=1&until_ms=2',self.calls[0]);receipt=json.loads(report.read_text());self.assertEqual(receipt['summary'],self.packet)
        suite=ET.parse(junit).getroot();self.assertEqual(suite.attrib['tests'],'2');self.assertEqual(suite.attrib['failures'],'0');self.assertEqual(len(suite.findall('testcase')),2);properties={item.get('name'):json.loads(item.get('value')) for item in suite.findall('properties/property')};self.assertEqual(properties['since_ms'],1);self.assertEqual(properties['until_ms'],2);self.assertEqual(properties['assessment_source'],'caller_reported');self.assertIs(properties['automatic_promotion'],False)
    def test_quality_and_error_failure_codes(self):
        junit=self.root/'failed.xml';code,_=self.run_cli('--callback-max-failed-assessments','0','--junit',str(junit));self.assertEqual(code,1);self.assertEqual(ET.parse(junit).getroot().attrib['failures'],'1')
        self.packet['metrics'][0]['evaluator_version']=1;code,_=self.run_cli();self.assertEqual(code,1)
        self.packet['provider_calls']=1;error_xml=self.root/'error.xml';code,_=self.run_cli('--junit',str(error_xml));self.assertEqual(code,2);self.assertEqual(ET.parse(error_xml).getroot().attrib['errors'],'1')
    def test_version_failure_limit_reports(self):
        self.config.write_text(json.dumps([dict(self.requirement,max_failed_assessments=0)]))
        self.packet['evaluators']=[dict(evaluator_id='check',evaluator_version=2,completed_assessments=2,failed_assessments=0),dict(evaluator_id='other',evaluator_version=1,completed_assessments=0,failed_assessments=1)]
        report=self.root/'version-pass.json';code,_=self.run_cli('--report',str(report));self.assertEqual(code,0)
        self.assertEqual(json.loads(report.read_text())['checks'][0]['evaluator_attempts']['failed_assessments'],0)
        self.packet['evaluators'][0]['failed_assessments']=1;self.packet['evaluators'].pop()
        junit=self.root/'version-fail.xml';code,_=self.run_cli('--junit',str(junit));self.assertEqual(code,1);self.assertIn('evaluator_failure_limit_exceeded',ET.parse(junit).getroot().find('testcase/failure').text)

    def test_preflight_no_network(self):
        for raw in ['[]','[{"evaluator_id":"a","evaluator_id":"b"}]','['+' '*131072+']']:
            self.config.write_text(raw);code,_=self.run_cli();self.assertEqual(code,2)
        self.assertEqual(self.calls,[])
    def test_existing_artifact_and_mixed_mode_rejected(self):
        report=self.root/'existing.json';report.write_text('preserve');code,_=self.run_cli('--report',str(report));self.assertEqual(code,2);self.assertEqual(report.read_text(),'preserve')
        for options in [('--provider','local'),('--min-score','quality=1'),('--concurrency','2'),('--task-async',)]:
            code,_=self.run_cli(*options);self.assertEqual(code,2)
        self.assertEqual(self.calls,[])
if __name__=='__main__':unittest.main()
