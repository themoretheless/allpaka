"""Explicit local input/output guardrails; no inference, retries or content capture."""
import hashlib
import math
import functools
import inspect
import json


class GuardrailBlocked(Exception):
    def __init__(self, receipt):
        super().__init__('guardrail_blocked')
        self.receipt = receipt


def _policy(rules):
    if not isinstance(rules, (list, tuple)) or not 1 <= len(rules) <= 32:
        raise ValueError('Use 1-32 explicit guardrail rules')
    frozen = []
    identifiers = set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {'id', 'kind', 'value'}:
            raise ValueError('Rule requires id, kind and value')
        identifier, kind, value = rule['id'], rule['kind'], rule['value']
        if (not isinstance(identifier, str) or not 1 <= len(identifier) <= 80
                or any(not (c.isascii() and (c.isalnum() or c in '._-')) for c in identifier)
                or identifier in identifiers):
            raise ValueError('Use unique bounded technical rule IDs')
        identifiers.add(identifier)
        if kind in ('min_bytes', 'max_bytes'):
            if type(value) is not int or not 0 <= value <= 64000:
                raise ValueError('Byte limit must be an integer in 0-64000')
        elif kind in ('forbidden_substrings', 'required_substrings'):
            if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 100
                    or any(not isinstance(x, str) or not 1 <= len(x.encode('utf-8')) <= 1000 for x in value)):
                raise ValueError('Use 1-100 nonempty bounded literal fragments')
            value = tuple(value)
        elif kind == 'json_valid':
            if value is not True:
                raise ValueError('JSON validation must explicitly be true')
        else:
            raise ValueError('Unsupported guardrail kind')
        frozen.append((identifier, kind, value))
    return tuple(frozen)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError('Nonfinite JSON number')


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite JSON number')
    return number


def _check(text, policy, stage, action):
    if not isinstance(text, str):
        raise ValueError('Guardrails require a text value')
    size = len(text.encode('utf-8'))
    if size > 64000:
        raise ValueError('Guardrail text exceeds 64000 UTF-8 bytes')
    results = []
    for identifier, kind, value in policy:
        if kind == 'min_bytes':
            passed = size >= value
        elif kind == 'max_bytes':
            passed = size <= value
        elif kind == 'forbidden_substrings':
            passed = not any(fragment in text for fragment in value)
        elif kind == 'required_substrings':
            passed = all(fragment in text for fragment in value)
        else:
            try:
                json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite, parse_float=_finite_float)
                passed = True
            except (ValueError, RecursionError):
                passed = False
        results.append(dict(rule_id=identifier, kind=kind, passed=passed))
    passed = all(x['passed'] for x in results)
    receipt = dict(kind='local_guardrail', schema_version=1, stage=stage, action=action,
                   passed=passed, blocked=not passed and action == 'block', rules=results,
                   policy_sha256=hashlib.sha256(json.dumps(policy,ensure_ascii=False,separators=(',', ':')).encode('utf-8')).hexdigest(),
                   provider_calls=0, content_captured=False)
    if receipt['blocked']:
        raise GuardrailBlocked(receipt)
    return receipt


def _traced_check(text, policy, stage, action, trace):
    try:
        receipt = _check(text, policy, stage, action)
        blocked = None
    except GuardrailBlocked as error:
        receipt, blocked = error.receipt, error
    if trace is not None:
        # Only technical outcome/configuration fingerprints enter the native tree.
        name = 'guardrail.%s.%s.%s.%s' % (stage, action,
                'pass' if receipt['passed'] else 'fail', receipt['policy_sha256'])
        with trace.span(name, 'tool') as span:
            span.set_guardrail_receipt(receipt)
            if blocked is not None:
                raise blocked
    elif blocked is not None:
        raise blocked
    return receipt


def check_guardrails(text, rules, *, stage='input', action='observe', trace=None):
    """Return metadata-only evidence or raise GuardrailBlocked with its receipt.

    Literal checks are case-sensitive; JSON rejects duplicate keys/nonfinite values.
    Limits count UTF-8 bytes. Optional active trace records only stage/action/outcome/hash.
    This does not claim moderation or prompt-injection detection.
    """
    if stage not in ('input', 'output') or action not in ('observe', 'block'):
        raise ValueError('Invalid guardrail stage or action')
    return _traced_check(text, _policy(rules), stage, action, trace)


def guard_task(task, *, input_rules, output_rules, action='block', trace=None, return_receipts=True):
    """Wrap a sync/async text task with frozen policies and explicit check receipts.

    Returns {'output': original_text, 'guardrails': [input_receipt, output_receipt]}.
    Explicit return_receipts=False returns text for Studio evaluate_task integration.
    A blocked input never executes the task. A blocked output is not returned.
    Business exceptions/cancellation propagate unchanged; the task is never retried.
    """
    if not callable(task) or action not in ('observe', 'block') or type(return_receipts) is not bool:
        raise ValueError('Use a callable and explicit guardrail action')
    before, after = _policy(input_rules), _policy(output_rules)
    def begin(text):
        return _traced_check(text, before, 'input', action, trace)
    def finish(output, receipt):
        try:
            checked = _traced_check(output, after, 'output', action, trace)
        except GuardrailBlocked as error:
            error.input_receipt = receipt
            raise
        return dict(output=output, guardrails=[receipt, checked]) if return_receipts else output
    if inspect.iscoroutinefunction(task) or inspect.iscoroutinefunction(getattr(task, '__call__', None)):
        @functools.wraps(task)
        async def asynchronous(text, *args, **kwargs):
            receipt = begin(text)
            return finish(await task(text, *args, **kwargs), receipt)
        asynchronous._allpaka_trace_span_budget = _span_budget(task, trace)
        return asynchronous
    @functools.wraps(task)
    def synchronous(text, *args, **kwargs):
        receipt = begin(text)
        return finish(task(text, *args, **kwargs), receipt)
    synchronous._allpaka_trace_span_budget = _span_budget(task, trace)
    return synchronous


def _span_budget(task, trace):
    nested = getattr(task, '_allpaka_trace_span_budget', None)
    if trace is None:
        return nested
    if isinstance(nested, tuple) and len(nested) == 2 and nested[0] is trace:
        return (trace, nested[1] + 2)
    return (trace, 2)


def policy_manifest(rules):
    """Return a validated, portable policy with its execution fingerprint."""
    frozen = _policy(rules)
    encoded = json.dumps(frozen, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > 128 * 1024:
        raise ValueError('Policy manifest exceeds 128 KiB')
    return dict(kind='local_guardrail_policy', schema_version=1,
                policy_sha256=hashlib.sha256(encoded).hexdigest(),
                rules=[dict(id=identifier, kind=kind,
                            value=list(value) if isinstance(value, tuple) else value)
                       for identifier, kind, value in frozen])


def rules_from_manifest(manifest):
    """Validate stored policy schema and fingerprint before returning fresh rules."""
    if (not isinstance(manifest, dict)
            or set(manifest) != {'kind', 'schema_version', 'policy_sha256', 'rules'}
            or manifest['kind'] != 'local_guardrail_policy'
            or type(manifest['schema_version']) is not int
            or manifest['schema_version'] != 1):
        raise ValueError('Invalid policy manifest schema')
    checked = policy_manifest(manifest['rules'])
    if manifest['policy_sha256'] != checked['policy_sha256']:
        raise ValueError('Policy manifest fingerprint mismatch')
    return checked['rules']


def save_policy(path, rules):
    """Create a private policy file atomically; never overwrite an existing path."""
    import os
    import tempfile
    from pathlib import Path
    manifest = policy_manifest(rules)
    payload = json.dumps(manifest, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(payload) > 128 * 1024:
        raise ValueError('Policy file exceeds 128 KiB')
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.allpaka-policy-', dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, str(destination))
    finally:
        os.unlink(temporary)
    return manifest


def load_policy(path):
    """Read a bounded regular policy file and validate its schema and hash."""
    import os
    import stat
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Policy file must be a regular file')
        payload = stream.read(128 * 1024 + 1)
    if len(payload) > 128 * 1024:
        raise ValueError('Policy file exceeds 128 KiB')
    manifest = json.loads(payload.decode('utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_nonfinite, parse_float=_finite_float)
    return rules_from_manifest(manifest)


def _trace_receipt(receipt):
    fields={'kind','schema_version','stage','action','passed','blocked','rules','policy_sha256','provider_calls','content_captured'}
    if (not isinstance(receipt,dict) or set(receipt)!=fields
            or receipt['kind']!='local_guardrail' or type(receipt['schema_version']) is not int or receipt['schema_version']!=1
            or receipt['stage'] not in ('input','output') or receipt['action'] not in ('observe','block')
            or type(receipt['passed']) is not bool or type(receipt['blocked']) is not bool
            or receipt['blocked'] != (not receipt['passed'] and receipt['action']=='block')
            or type(receipt['provider_calls']) is not int or receipt['provider_calls']!=0 or receipt['content_captured'] is not False
            or not isinstance(receipt['policy_sha256'],str) or len(receipt['policy_sha256'])!=64
            or any(c not in '0123456789abcdef' for c in receipt['policy_sha256'])):
        raise ValueError('Invalid guardrail trace receipt')
    rules=receipt['rules']
    if not isinstance(rules,list) or not 1<=len(rules)<=32:
        raise ValueError('Invalid guardrail rule outcomes')
    identifiers=set()
    for rule in rules:
        if (not isinstance(rule,dict) or set(rule)!={'rule_id','kind','passed'}
                or not isinstance(rule['rule_id'],str) or not 1<=len(rule['rule_id'])<=80
                or any(not(c.isascii() and (c.isalnum() or c in '._-')) for c in rule['rule_id'])
                or rule['rule_id'] in identifiers or type(rule['passed']) is not bool
                or rule['kind'] not in ('min_bytes','max_bytes','json_valid','forbidden_substrings','required_substrings')):
            raise ValueError('Invalid guardrail rule outcome')
        identifiers.add(rule['rule_id'])
    if all(rule['passed'] for rule in rules)!=receipt['passed']:
        raise ValueError('Inconsistent guardrail receipt')
    return json.loads(json.dumps(receipt))
