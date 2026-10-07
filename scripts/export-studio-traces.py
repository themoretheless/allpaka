#!/usr/bin/env python3
"""Export explicitly selected Studio traces to a new JSON file; no model calls."""
import argparse
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'sdk/python'))
from allpaka_studio import Studio, EvaluationError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--project-id',default='default')
    parser.add_argument('--trace-id',action='append',required=True,help='Selected trace ID; repeat up to 100 times')
    parser.add_argument('--include-feedback',action='store_true',help='Explicitly include manually entered feedback text')
    parser.add_argument('--output',required=True,help='New export file; existing files are never replaced')
    parser.add_argument('--format',choices=('json','jsonl'),default='json',help='Full batch JSON or one full trace export packet per JSONL line')
    args=parser.parse_args()
    target=pathlib.Path(args.output).absolute()
    temporary=None
    try:
        if target.exists() or target.is_symlink():raise FileExistsError()
        packet=Studio(args.base_url).export_traces(args.project_id,args.trace_id,include_feedback=args.include_feedback)
        items=packet.get('traces',[])
        if (packet.get('kind')!='trace_export_batch' or packet.get('schema_version')!=1
                or packet.get('provider_calls')!=0
                or packet.get('project_id')!=args.project_id or packet.get('trace_count')!=len(args.trace_id)
                or not isinstance(items,list) or len(items)!=len(args.trace_id)
                or [item.get('trace',{}).get('id') for item in items]!=args.trace_id
                or any(item.get('trace',{}).get('project_id')!=args.project_id for item in items)
                or packet.get('privacy',{}).get('feedback_included') is not args.include_feedback
                or (not args.include_feedback and any(item.get('feedback') is not None for item in items))):
            raise EvaluationError('invalid_export_packet')
        if args.format=='jsonl':
            data=(''.join(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n' for item in items)).encode('utf-8')
        else:
            data=(json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
        target.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix='.trace-export-',dir=target.parent)
        with os.fdopen(descriptor,'wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.link(temporary,target)
        print(json.dumps(dict(saved=True,format=args.format,trace_count=len(items),include_feedback=args.include_feedback,provider_calls=0)))
        return 0
    except KeyboardInterrupt:
        print(json.dumps(dict(saved=False,error='interrupted')));return 130
    except (EvaluationError,ValueError,OSError,TypeError,AttributeError):
        print(json.dumps(dict(saved=False,error='export_not_saved')));return 2
    finally:
        if temporary is not None:
            try:os.unlink(temporary)
            except FileNotFoundError:pass


if __name__=='__main__':sys.exit(main())
