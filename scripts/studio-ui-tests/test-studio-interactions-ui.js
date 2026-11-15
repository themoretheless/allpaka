const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

function makeNode(tag) {
  const el = {
    tagName: String(tag || 'div').toUpperCase(),
    children: [],
    className: '',
    hidden: false,
    textContent: '',
    value: '',
    checked: false,
    type: '',
    name: '',
    placeholder: '',
    attributes: {},
    setAttribute(k, v) { this.attributes[k] = String(v); },
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = [...nodes]; },
    addEventListener() {},
  };
  return el;
}

const harnessSrc = fs.readFileSync('crates/allpaka-chat/web/harness-ext.js', 'utf8');
const els = {
  'interaction-banner': makeNode('section'),
  'session-cwd-badge': makeNode('span'),
};
const posted = [];
const context = {
  document: {
    getElementById: (id) => els[id] || (els[id] = makeNode('div')),
    createElement: (tag) => makeNode(tag),
    createTextNode: (t) => ({ nodeType: 3, textContent: String(t) }),
    querySelectorAll: () => [],
  },
};
vm.createContext(context);
const ext = vm.runInContext(`${harnessSrc}\n;StudioHarnessExt;`, context);

ext.init({
  $: (id) => els[id] || (els[id] = makeNode('div')),
  api: async (path, body) => {
    posted.push({ path, body });
    return { accepted: true };
  },
  del: async () => ({}),
  node: (tag, text, cls) => {
    const n = makeNode(tag);
    if (text !== undefined) n.textContent = String(text);
    if (cls) n.className = cls;
    return n;
  },
  notify: () => {},
  fail: (e) => { throw e; },
  getSessionId: () => 'ses_1',
  getCurrent: () => null,
  getProjectId: () => 'default',
  reloadSession: async () => {},
});

assert.deepEqual(ext.parsePendingInteraction(null), { kind: null, approval: null, question: null });

const approvalPayload = ext.buildApprovalPayload('req_a1', 'approve_once', 'safe edit');
assert.deepEqual(approvalPayload, {
  request_id: 'req_a1',
  kind: 'approval',
  decision: 'approve_once',
  reason: 'safe edit',
});

assert.throws(() => ext.buildApprovalPayload('req_a1', 'invented'), /Invalid approval decision/);
assert.deepEqual(ext.buildQuestionPayload('req_q1', { q1: 'yes' }), {
  request_id: 'req_q1',
  kind: 'question',
  answers: { q1: 'yes' },
});
assert.deepEqual(ext.buildQuestionPayload('req_q1', {}, 'cancelled'), {
  request_id: 'req_q1',
  kind: 'question',
  decision: 'dismiss',
  reason: 'cancelled',
});

ext.syncSession({
  id: 'ses_1',
  working_dir: 'crates/allpaka-chat',
  pending_approval: { id: 'apr_1', tool: 'write_file', summary: 'Write a.txt', args_preview: '{"path":"a.txt"}' },
});
assert.equal(els['session-cwd-badge'].hidden, false);
assert.equal(els['session-cwd-badge'].textContent, 'cwd: crates/allpaka-chat');
assert.equal(els['interaction-banner'].hidden, false);
assert(els['interaction-banner'].children.length >= 2);

ext.syncSession({ id: 'ses_1', working_dir: '.' });
assert.equal(els['session-cwd-badge'].hidden, true);
assert.equal(els['interaction-banner'].hidden, true);

ext.syncSession({
  id: 'ses_1',
  pending_question: {
    id: 'qst_1',
    questions: [{ id: 'target', question: 'Select target?', options: ['mac', 'linux'], multi_select: false }],
  },
});
assert.equal(els['interaction-banner'].hidden, false);

ext.reset();
assert.equal(els['interaction-banner'].hidden, true);

console.log('PASS Studio pending approval/question interaction parsing, payloads and banner state');
