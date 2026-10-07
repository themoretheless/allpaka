#!/usr/bin/env python3
"""CI entry point for Studio evaluations and explicitly selected local Python tasks."""
import asyncio
import runpy
from contextlib import nullcontext
import argparse
import json
import os
import pathlib
import signal
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sdk/python"))
from allpaka_studio import EvaluationError, Studio
from allpaka_guardrails import GuardrailBlocked, guard_task, load_policy


def guardrail_evidence(error):
    if not isinstance(error, GuardrailBlocked):return None
    receipt=error.receipt
    if not isinstance(receipt,dict):return None
    fingerprint=receipt.get('policy_sha256')
    if (receipt.get('stage') not in ('input','output') or receipt.get('action')!='block'
            or receipt.get('passed') is not False or receipt.get('blocked') is not True
            or not isinstance(fingerprint,str) or len(fingerprint)!=64
            or any(c not in '0123456789abcdef' for c in fingerprint)):
        return None
    return dict(kind='local_guardrail',schema_version=1,stage=receipt['stage'],
                action='block',passed=False,blocked=True,policy_sha256=fingerprint,
                provider_calls=0,content_captured=False)


def load_guardrails(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate guardrail configuration key')
            result[key]=value
        return result
    with open(path,'rb') as handle:raw=handle.read(128*1024+1)
    if len(raw)>128*1024:raise ValueError('Guardrail configuration exceeds 128 KiB')
    config=json.loads(raw,object_pairs_hook=unique)
    if (not isinstance(config,dict) or set(config)!={'schema_version','action','input_rules','output_rules'}
            or type(config['schema_version']) is not int or config['schema_version']!=1):
        raise ValueError('Invalid guardrail configuration schema')
    options=dict(input_rules=config['input_rules'],output_rules=config['output_rules'],action=config['action'])
    guard_task(lambda text:text,**options)  # Validate before loading/executing task code.
    return options


def load_policy_guardrails(input_path, output_path, action):
    options=dict(input_rules=load_policy(input_path),output_rules=load_policy(output_path),action=action)
    guard_task(lambda text:text,**options)
    return options


def artifact(path, data):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".evaluation-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Retained evidence must not silently replace another run's report.
        os.link(temporary, target)
    finally:
        os.unlink(temporary)


def task_junit(result, summary, code):
    """Per-sample diagnostics plus one aggregate quality gate, without answer text."""
    result=result or {}
    evaluated=result if result.get('receipt') else result.get('evaluation_result') or {}
    receipt=evaluated.get('receipt') or {}
    items=receipt.get('items') or []
    execution_error=code not in (0,1)
    partial=result.get('completed_outputs') or {}
    rows=[(item['sample_id'],item.get('scores'),False) for item in items]
    if not items:rows=[(sample_id,None,True) for sample_id in partial]
    has_gate=bool(receipt)
    failed_gate=has_gate and not evaluated.get('passed',False)
    suite=ET.Element('testsuite',name='allpaka-python-task',
        tests=str(len(rows)+int(has_gate)+int(execution_error)),
        failures=str(int(failed_gate)),errors=str(int(execution_error)),
        skipped=str(sum(skipped for _,_,skipped in rows)))
    for sample_id,scores,skipped in rows:
        case=ET.SubElement(suite,'testcase',classname='allpaka.studio.task.sample',name=sample_id)
        if skipped:ET.SubElement(case,'skipped',message='callback_completed_score_unavailable')
        ET.SubElement(case,'system-out').text=json.dumps(dict(sample_id=sample_id,
            callback_completed=True,scores=scores,receipt_id=receipt.get('id')),allow_nan=False)
    if has_gate:
        case=ET.SubElement(suite,'testcase',classname='allpaka.studio.task',name='quality_gate')
        evidence=dict(receipt_id=receipt['id'],passed=evaluated.get('passed',False),
            mean_scores=receipt.get('mean_scores'),thresholds=evaluated.get('thresholds'),
            comparison_id=(evaluated.get('comparison') or {}).get('id'))
        if failed_gate:ET.SubElement(case,'failure',message='quality_gate_failed').text=json.dumps(evidence)
        ET.SubElement(case,'system-out').text=json.dumps(evidence)
    if execution_error:
        sample_id=result.get('sample_id')
        case=ET.SubElement(suite,'testcase',classname='allpaka.studio.task',
            name='execution.'+sample_id if sample_id else 'execution')
        evidence=dict(error=summary.get('error','execution_failed'),sample_id=sample_id,
            evaluation_phase=result.get('evaluation_phase'),trace_id=summary.get('trace_id'),
            trace_export_failed=summary.get('trace_export_failed',False),
            guardrail=result.get('guardrail'))
        ET.SubElement(case,'error',message=evidence['error']).text=json.dumps(evidence)
    return suite


def callback_summary_ci(args):
    outputs=[pathlib.Path(p).absolute() for p in (args.report,args.junit) if p]
    if len(set(outputs))!=len(outputs) or any(p.exists() for p in outputs):
        print(json.dumps(dict(passed=False,error='artifact_path_exists_or_duplicate')));return 2
    result=None
    try:
        def unique(pairs):
            value={}
            for key,item in pairs:
                if key in value:raise ValueError('Duplicate callback requirement key')
                value[key]=item
            return value
        with open(args.callback_requirements_file,'rb') as handle:raw=handle.read(128*1024+1)
        if len(raw)>128*1024:raise ValueError('Callback requirements exceed 128 KiB')
        requirements=json.loads(raw,object_pairs_hook=unique)
        result=Studio(args.base_url).check_callback_evaluation_summary(args.project_id,requirements,
            since_ms=args.callback_since_ms,until_ms=args.callback_until_ms,
            max_failed_assessments=args.callback_max_failed_assessments)
        code=0 if result['passed'] else 1
        summary=dict(kind=result['kind'],passed=result['passed'],project_id=args.project_id,
            assessment_source='caller_reported',provider_calls=0,automatic_promotion=False)
    except KeyboardInterrupt:
        code=130;summary=dict(passed=False,error='interrupted')
    except Exception as error:
        code=2;summary=dict(passed=False,error=error.reason if isinstance(error,EvaluationError) else 'invalid_request_or_receipt')
    try:
        if args.report:artifact(args.report,(json.dumps(result or summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
        if args.junit:
            checks=[] if result is None else result['checks']
            failure=None if result is None else result['failure_check']
            suite=ET.Element('testsuite',name='allpaka-callback-evaluation',tests=str(len(checks)+int(failure is not None)+int(code not in (0,1))),failures=str(sum(not check['passed'] for check in checks)+int(failure is not None and not failure['passed'])),errors=str(int(code not in (0,1))))
            properties=ET.SubElement(suite,'properties')
            scope=dict(project_id=args.project_id,since_ms=args.callback_since_ms,until_ms=args.callback_until_ms,assessment_source='caller_reported',provider_calls=0,automatic_promotion=False)
            for key,value in scope.items():ET.SubElement(properties,'property',name=key,value=json.dumps(value,ensure_ascii=False,allow_nan=False))
            for index,check in enumerate(checks):
                requirement=check['requirement'];case=ET.SubElement(suite,'testcase',classname='allpaka.studio.callback',name=str(index)+'.'+requirement['evaluator_id']+'.v'+str(requirement['evaluator_version'])+'.'+requirement['metric'])
                if not check['passed']:ET.SubElement(case,'failure',message='callback_threshold_failed').text=json.dumps(check)
                ET.SubElement(case,'system-out').text=json.dumps(check)
            if failure is not None:
                case=ET.SubElement(suite,'testcase',classname='allpaka.studio.callback',name='entire_summary.failed_assessments')
                if not failure['passed']:ET.SubElement(case,'failure',message='callback_failure_limit_exceeded').text=json.dumps(failure)
                ET.SubElement(case,'system-out').text=json.dumps(failure)
            if code not in (0,1):
                case=ET.SubElement(suite,'testcase',classname='allpaka.studio.callback',name='execution');ET.SubElement(case,'error',message=summary['error']).text=json.dumps(summary)
            artifact(args.junit,ET.tostring(suite,encoding='utf-8',xml_declaration=True))
    except OSError:
        code=2;summary['artifact_error']='report_not_saved'
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False));return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project-id", default="default")
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--callback-requirements-file", help="JSON list of version-pinned callback metric requirements; checks saved telemetry without model calls")
    prompt.add_argument("--prompt-id")
    prompt.add_argument("--prompt-template")
    prompt.add_argument("--outputs-file", help="JSON mapping sample IDs to ready answers; scores without model calls")
    prompt.add_argument("--judge-plan-id", help="Run a saved model-judge batch plan")
    prompt.add_argument("--matrix-file", help="JSON object with requests and optional labels for 2–16 variants")
    prompt.add_argument("--task-file", help="Execute a trusted local Python file and evaluate its callback")
    prompt.add_argument("--scored-judge-id", help="Combine saved deterministic scores with explicit model judging")
    prompt.add_argument("--review-score-id", help="Recheck saved scores and a completed judge run without model calls")
    parser.add_argument("--review-judge-run-id")
    parser.add_argument("--judge-preset")
    parser.add_argument("--judge-preset-version", type=int)
    parser.add_argument("--judge-baseline-id")
    parser.add_argument("--require-judge-improvement", action="store_true")
    parser.add_argument("--guardrails-file", help="Explicit bounded JSON input/output guardrails for a Python task")
    parser.add_argument("--input-policy-file", help="Fingerprint-verified input policy manifest")
    parser.add_argument("--output-policy-file", help="Fingerprint-verified output policy manifest")
    parser.add_argument("--policy-action", choices=["block","observe"], help="Explicit action for policy manifest pair")
    parser.add_argument("--task-function", default="task")
    parser.add_argument("--task-async", action="store_true")
    parser.add_argument("--with-contexts", action="store_true")
    parser.add_argument("--trace-correlation", help="Record task metadata in the selected project with this technical correlation ID")
    parser.add_argument("--trace-key", help="Optional idempotency key selected before task execution")
    parser.add_argument("--judge-min-score", type=float, help="Required mean rubric score for a judge plan")
    parser.add_argument("--prompt-version", type=int)
    parser.add_argument("--metric", action="append", choices=["exact_match", "contains_reference", "json_valid", "whitespace_token_f1", "json_equals", "character_bigram_f1"])
    parser.add_argument("--min-score", action="append", default=[], help="Required mean score, e.g. exact_match=1; repeat for multiple metrics")
    parser.add_argument("--baseline-id")
    parser.add_argument("--persist-matrix", action="store_true", help="Save the completed matrix evidence in Studio (requires --matrix-file)")
    parser.add_argument("--require-improvement", action="store_true", help="Also reject ties; requires --baseline-id")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--item-timeout", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--report", help="New JSON file with full run and comparison evidence")
    parser.add_argument("--junit", help="New JUnit XML file for CI")
    parser.add_argument('--callback-since-ms',type=int)
    parser.add_argument('--callback-until-ms',type=int)
    parser.add_argument('--callback-max-failed-assessments',type=int)
    args = parser.parse_args()
    callback_options=('callback_since_ms','callback_until_ms','callback_max_failed_assessments')
    if args.callback_requirements_file:
        allowed={'base_url','project_id','callback_requirements_file','report','junit',*callback_options}
        if any(value!=parser.get_default(key) for key,value in vars(args).items() if key not in allowed):
            parser.error('Callback summary checks accept project, time boundaries, failure limit and report options only')
        return callback_summary_ci(args)
    if any(getattr(args,key) is not None for key in callback_options):
        parser.error('Callback options require --callback-requirements-file')
    if args.persist_matrix and not args.matrix_file:
        parser.error("--persist-matrix requires --matrix-file")
    if args.matrix_file:
        if (args.dataset_id or args.dataset_version is not None or args.provider or args.model or args.metric
                or args.prompt_version is not None or args.baseline_id or args.project_id != 'default'
                or args.concurrency != 4 or args.item_timeout != 120):
            parser.error("Matrix files pin all variants; do not mix dataset/model/baseline flags")
    elif args.review_score_id:
        if not args.review_judge_run_id:
            parser.error('--review-score-id requires --review-judge-run-id')
        if (args.dataset_id or args.dataset_version is not None or args.metric or args.prompt_version is not None
                or args.provider or args.model or args.baseline_id or args.require_improvement or args.project_id != 'default'
                or args.concurrency != 4 or args.item_timeout != 120):
            parser.error('Saved review accepts receipt IDs and thresholds only')
    elif args.scored_judge_id:
        if args.dataset_id or args.dataset_version is not None or args.metric or args.prompt_version is not None:
            parser.error('Saved score receipt already pins dataset and metrics')
        if not args.judge_preset or args.judge_preset_version is None or args.judge_preset_version<1:
            parser.error('Combined evaluation requires a pinned judge preset')
    elif args.judge_plan_id:
        if args.dataset_id or args.dataset_version is not None or args.provider or args.model or args.metric or args.min_score or args.prompt_version is not None:
            parser.error("Judge plans already pin source/model/rubric; do not mix evaluation modes")
    elif not args.dataset_id or args.dataset_version is None or not args.metric:
        parser.error("Dataset evaluation requires --dataset-id, --dataset-version and --metric")
    if args.review_judge_run_id and not args.review_score_id:
        parser.error('--review-judge-run-id requires --review-score-id')
    if not args.scored_judge_id and (args.judge_preset or args.judge_preset_version is not None or args.judge_baseline_id or args.require_judge_improvement):
        parser.error("Combined judge options require --scored-judge-id")
    if args.require_judge_improvement and not args.judge_baseline_id:
        parser.error("--require-judge-improvement requires --judge-baseline-id")
    if args.judge_min_score is not None and not (args.judge_plan_id or args.scored_judge_id or args.review_score_id):
        parser.error("--judge-min-score requires --judge-plan-id")
    if any((args.input_policy_file,args.output_policy_file,args.policy_action)):
        if not all((args.input_policy_file,args.output_policy_file,args.policy_action)):
            parser.error("Policy manifests require input/output files and explicit --policy-action")
        if args.guardrails_file:
            parser.error("Do not mix --guardrails-file with policy manifests")
    if not args.task_file and (args.input_policy_file or args.output_policy_file or args.policy_action or args.guardrails_file or args.task_async or args.with_contexts or args.task_function != "task" or args.trace_correlation or args.trace_key):
        parser.error("Task options require --task-file")
    if args.trace_key and not args.trace_correlation:
        parser.error("--trace-key requires --trace-correlation")
    if args.task_file and not args.task_async and (args.concurrency != 4 or args.item_timeout != 120):
        parser.error("Concurrency and item timeout require --task-async")
    if args.task_file and not args.task_function.isidentifier():
        parser.error("Choose a Python function identifier")
    if args.outputs_file or args.task_file:
        if args.provider or args.model or args.prompt_version is not None:
            parser.error("Offline scoring does not accept model, provider or prompt version")
    elif not args.judge_plan_id and not args.matrix_file and not args.task_file and not args.review_score_id and (not args.provider or not args.model):
        parser.error("Provider evaluations require --provider and --model")
    if bool(args.prompt_id) != (args.prompt_version is not None):
        parser.error("--prompt-id requires --prompt-version; inline prompts do not use a version")
    if args.require_improvement and not args.baseline_id and not args.matrix_file:
        parser.error("--require-improvement requires --baseline-id")
    if (args.dataset_version is not None and args.dataset_version < 1) or (args.prompt_version is not None and args.prompt_version < 1):
        parser.error("Snapshot versions must be positive")
    minimums = {}
    for threshold in args.min_score:
        try:
            metric, value = threshold.split("=", 1)
            if metric in minimums or (not args.matrix_file and not args.scored_judge_id and not args.review_score_id and metric not in args.metric): raise ValueError()
            minimums[metric] = float(value)
        except ValueError:
            parser.error("Use unique --min-score METRIC=NUMBER thresholds for selected metrics")
    outputs = [pathlib.Path(p).absolute() for p in (args.report, args.junit) if p]
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        print(json.dumps(dict(passed=False, error="artifact_path_exists_or_duplicate")))
        return 2
    request = dict(dataset_id=args.dataset_id, dataset_version=args.dataset_version,
                   settings=dict(project_id=args.project_id, provider=args.provider, model=args.model, mode="chat", allow_writes=False),
                   metrics=sorted(set(args.metric or [])), concurrency=args.concurrency, item_timeout_secs=args.item_timeout)
    if args.prompt_id:
        request["prompt_ref"] = dict(id=args.prompt_id, version=args.prompt_version)
    else:
        request["prompt_template"] = args.prompt_template
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    result = None
    try:
        if args.matrix_file:
            def unique_matrix_object(pairs):
                value = {}
                for key, item in pairs:
                    if key in value: raise ValueError("Duplicate matrix JSON key")
                    value[key] = item
                return value
            with open(args.matrix_file, 'rb') as handle:
                raw = handle.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024: raise ValueError("Matrix file too large")
            matrix = json.loads(raw, object_pairs_hook=unique_matrix_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite matrix value")))
            if not isinstance(matrix, dict) or set(matrix) - {'requests', 'labels'}:
                raise ValueError("Matrix file requires requests and optional labels")
            result = Studio(args.base_url).evaluate_matrix(matrix['requests'], labels=matrix.get('labels'),
                timeout=args.timeout, min_scores=minimums, require_improvement=args.require_improvement, persist=args.persist_matrix)
            passed = all(row['result']['passed'] for row in result['variants'])
            code = 0 if passed else 1
            result['passed'] = passed
            summary = dict(kind=result['kind'], passed=passed, baseline_id=result['baseline_id'], native_matrix_id=result.get('native_matrix',{}).get('id'),
                           variants=[dict(label=row['label'], run_id=row['result']['run']['id'],
                                          passed=row['result']['passed'], comparison=row['result']['comparison'])
                                     for row in result['variants']])
        elif args.review_score_id:
            result=Studio(args.base_url).review_scored_output_judges(args.review_score_id,args.review_judge_run_id,
                min_scores=minimums,min_judge_score=args.judge_min_score)
            code=0 if result['passed'] else 1
            summary=dict(kind=result['kind'],passed=result['passed'],score_id=args.review_score_id,
                judge_run_id=args.review_judge_run_id,deterministic_passed=result['deterministic']['passed'],
                judge_passed=result['judge']['passed'],mean_scores=result['deterministic']['receipt']['mean_scores'],
                mean_judge_score=result['judge']['observed'],provider_calls=0,automatic_promotion=False)
        elif args.scored_judge_id:
            result=Studio(args.base_url).evaluate_scored_output_judges(args.scored_judge_id,request['settings'],
                judge_preset={'id':args.judge_preset,'version':args.judge_preset_version},min_scores=minimums,
                min_judge_score=args.judge_min_score,baseline_score_id=args.baseline_id,
                baseline_judge_id=args.judge_baseline_id,require_improvement=args.require_improvement,
                require_judge_improvement=args.require_judge_improvement,timeout=args.timeout)
            code=0 if result['passed'] else 1
            summary=dict(kind=result['kind'],passed=result['passed'],score_id=args.scored_judge_id,
                judge_run_id=result['judge']['run']['id'],deterministic_passed=result['deterministic']['passed'],
                judge_passed=result['judge']['passed'],mean_scores=result['deterministic']['receipt']['mean_scores'],
                mean_judge_score=result['judge']['observed'],automatic_promotion=False)
        elif args.judge_plan_id:
            result=Studio(args.base_url).evaluate_judge_plan(args.judge_plan_id,min_score=args.judge_min_score,baseline_id=args.baseline_id,require_improvement=args.require_improvement,timeout=args.timeout)
            run=result["run"]
            code=0 if result["passed"] else 1 if run["status"]=="completed" else 2
            summary=dict(run_id=run["id"],kind="judge_run",status=run["status"],passed=result["passed"],
                         plan_id=run["plan_id"],plan_sha256=run["plan_sha256"],trace_id=run["trace_id"],
                         dataset_sha256=run["dataset_sha256"],mean_score=result["observed"],min_score=args.judge_min_score)
            if result.get('comparison') is not None:
                summary.update(comparison_id=result['comparison']['id'],regressions=result['comparison']['regressions'],improvements=result['comparison']['improvements'])
        elif args.task_file:
            guardrails=(load_guardrails(args.guardrails_file) if args.guardrails_file else
                        load_policy_guardrails(args.input_policy_file,args.output_policy_file,args.policy_action) if args.input_policy_file else None)
            path=pathlib.Path(args.task_file).resolve()
            if not path.is_file() or path.stat().st_size>1024*1024:
                raise ValueError('Task source must be a local file within 1 MiB')
            task=runpy.run_path(str(path))[args.task_function]
            client=Studio(args.base_url)
            options=dict(min_scores=minimums,baseline_id=args.baseline_id,
                         require_improvement=args.require_improvement,with_contexts=args.with_contexts)
            task_trace=client.trace(args.project_id,args.trace_correlation,idempotency_key=args.trace_key) if args.trace_correlation else None
            if guardrails is not None:task=guard_task(task,**guardrails,trace=task_trace,return_receipts=False)
            with task_trace if task_trace is not None else nullcontext():
                if task_trace is not None:options['trace']=task_trace
                if args.task_async:
                    result=asyncio.run(client.evaluate_task_async(args.dataset_id,args.dataset_version,task,
                        request['metrics'],item_timeout=args.item_timeout,concurrency=args.concurrency,**options))
                else:
                    result=client.evaluate_task(args.dataset_id,args.dataset_version,task,request['metrics'],**options)
            receipt=result['receipt'];code=0 if result['passed'] else 1
            summary=dict(kind='python_task_evaluation',receipt_id=receipt['id'],passed=result['passed'],
                dataset_sha256=receipt['dataset_sha256'],mean_scores=receipt['mean_scores'],
                gate_policy=result['gate_policy'],thresholds=result['thresholds'],comparison=result['comparison'])
        elif args.outputs_file:
            def unique_object(pairs):
                value = {}
                for key, item in pairs:
                    if key in value: raise ValueError("Duplicate output sample ID")
                    value[key] = item
                return value
            with open(args.outputs_file, "rb") as handle:
                raw = handle.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024: raise ValueError("Outputs file too large")
            ready = json.loads(raw, object_pairs_hook=unique_object)
            if not isinstance(ready, dict) or not ready or any(not isinstance(value, str) for value in ready.values()):
                raise ValueError("Outputs must map sample IDs to answer strings")
            result = Studio(args.base_url).evaluate_outputs(args.dataset_id, args.dataset_version, ready,
                       request["metrics"], min_scores=minimums, baseline_id=args.baseline_id, require_improvement=args.require_improvement)
            receipt = result["receipt"]
            code = 0 if result["passed"] else 1
            summary = dict(receipt_id=receipt["id"], kind="offline_score", passed=result["passed"],
                       gate_policy=result["gate_policy"], thresholds=result["thresholds"],
                       dataset_sha256=receipt["dataset_sha256"], mean_scores=receipt["mean_scores"], comparison=result["comparison"])
        else:
            result = Studio(args.base_url).evaluate(request, args.baseline_id, timeout=args.timeout, require_improvement=args.require_improvement, min_scores=minimums)
            run = result["run"]
            code = 0 if result["passed"] else 1 if run["status"] == "completed" and run["strict_quality"] else 2
            summary = dict(run_id=run["id"], status=run["status"], passed=result["passed"], gate_policy=result["gate_policy"], thresholds=result["thresholds"],
                           dataset_sha256=run["dataset_sha256"], mean_scores=run["mean_scores"], comparison=result["comparison"])
    except KeyboardInterrupt as error:
        task_error = error
        code = 130
        summary = dict(passed=False, error="interrupted", run_id=getattr(error, "run_id", None), cancellation_failed=getattr(error, "cancellation_failed", False))
        if args.matrix_file:
            result = dict(summary, kind='client_experiment_matrix', variants=getattr(error, 'matrix_results', []),
                          variant_index=getattr(error, 'variant_index', None), automatic_promotion=False)
    except Exception as error:
        task_error = error
        code = 2
        summary = dict(passed=False, error=error.reason if isinstance(error, EvaluationError) else "guardrail_blocked" if isinstance(error,GuardrailBlocked) else "invalid_request_or_receipt",
                       run_id=getattr(error, "run_id", None), cancellation_failed=getattr(error, "cancellation_failed", False))
        if args.matrix_file:
            result = dict(summary, kind='client_experiment_matrix',
                          variants=getattr(error, 'matrix_results', []),
                          variant_index=getattr(error, 'variant_index', None),
                          variant_label=getattr(error, 'variant_label', None), automatic_promotion=False)
    if args.scored_judge_id and code not in (0,1):
        result=dict(summary,kind='client_combined_evaluation',automatic_promotion=False,
            deterministic=getattr(locals().get('task_error'),'deterministic_result',None),
            judge_plan_id=getattr(locals().get('task_error'),'judge_plan_id',None),
            judge_run_receipt=getattr(locals().get('task_error'),'judge_run_receipt',None),
            judge_evidence_unavailable=getattr(locals().get('task_error'),'judge_evidence_unavailable',False))
    if args.task_file and code not in (0,1):
        evaluation_result=result
        result=dict(summary,kind='python_task_evaluation',automatic_promotion=False,
            evaluation_result=evaluation_result,
            completed_outputs=getattr(locals().get('task_error'), 'completed_outputs', {}),
            evaluation_phase=getattr(locals().get('task_error'),'evaluation_phase',None),
            sample_id=getattr(locals().get('task_error'),'sample_id',None),
            started_sample_ids=getattr(locals().get('task_error'),'started_sample_ids',[]),
            guardrail=guardrail_evidence(locals().get('task_error')))
    task_trace=locals().get('task_trace')
    if task_trace is not None:
        trace_id=task_trace.receipt.get('id') if task_trace.receipt else None
        summary.update(trace_id=trace_id,trace_export_failed=task_trace.export_error is not None)
        if result is not None:result.update(trace_id=trace_id,trace_export_failed=task_trace.export_error is not None)
    try:
        if args.report:
            artifact(args.report, (json.dumps(result or summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode())
        if args.junit:
            if args.scored_judge_id or args.review_score_id:
                components=[(name,(result or {}).get(name)) for name in ('deterministic','judge')]
                components=[(name,value) for name,value in components if value is not None]
                execution_error=code not in (0,1)
                suite=ET.Element('testsuite',name='allpaka-combined',tests=str(len(components)+int(execution_error)),
                    failures=str(sum(not value['passed'] for _,value in components)),errors=str(int(execution_error)))
                for name,value in components:
                    case=ET.SubElement(suite,'testcase',classname='allpaka.studio.combined',name=name)
                    if not value['passed']:ET.SubElement(case,'failure',message='quality_gate_failed').text=json.dumps(value)
                    ET.SubElement(case,'system-out').text=json.dumps(value)
                if execution_error:
                    case=ET.SubElement(suite,'testcase',classname='allpaka.studio.combined',name='execution')
                    ET.SubElement(case,'error',message=summary['error']).text=json.dumps(summary)
            elif args.task_file:
                suite=task_junit(result,summary,code)
            elif args.matrix_file:
                rows = (result or {}).get('variants', [])
                failures = sum(not row['result']['passed'] for row in rows)
                execution_error = code not in (0, 1)
                suite = ET.Element('testsuite', name='allpaka-matrix', tests=str(len(rows)+int(execution_error)),
                                   failures=str(failures), errors=str(int(execution_error)))
                for row in rows:
                    case = ET.SubElement(suite, 'testcase', classname='allpaka.studio.matrix', name=row['label'])
                    evidence = dict(run_id=row['result']['run']['id'], passed=row['result']['passed'],
                                    comparison=row['result'].get('comparison'), thresholds=row['result'].get('thresholds'))
                    if not row['result']['passed']:
                        ET.SubElement(case, 'failure', message='quality_gate_failed').text=json.dumps(evidence)
                    ET.SubElement(case, 'system-out').text=json.dumps(evidence)
                if execution_error:
                    case = ET.SubElement(suite, 'testcase', classname='allpaka.studio.matrix', name='matrix_execution')
                    ET.SubElement(case, 'error', message=summary['error']).text=json.dumps(summary)
            else:
                suite = ET.Element("testsuite", name="allpaka-evaluation", tests="1", failures=str(int(code == 1)), errors=str(int(code not in (0, 1))))
                case = ET.SubElement(suite, "testcase", classname="allpaka.studio", name="judge_evaluation" if args.judge_plan_id else "paired_evaluation" if args.baseline_id else "dataset_evaluation")
                if code:
                    ET.SubElement(case, "failure" if code == 1 else "error", message=summary.get("error", "quality_gate_failed")).text=json.dumps(summary)
                ET.SubElement(case, "system-out").text=json.dumps(summary)
            artifact(args.junit, ET.tostring(suite, encoding="utf-8", xml_declaration=True))
    except OSError:
        code = 2
        summary["artifact_error"] = "report_not_saved"
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
