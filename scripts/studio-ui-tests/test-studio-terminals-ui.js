const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

function makeNode(tag) {
  return {
    tagName: String(tag || 'div').toUpperCase(),
    children: [],
    className: '',
    hidden: false,
    textContent: '',
    value: '',
    checked: false,
    dataset: {},
    classList: { toggle() {} },
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = [...nodes]; },
  };
}

const harnessSrc = fs.readFileSync('crates/allpaka-chat/web/harness-ext.js', 'utf8');
const els = {
  'ops-pty-cmd': Object.assign(makeNode('input'), { value: 'echo hi' }),
  'ops-pty-input': Object.assign(makeNode('input'), { value: 'pwd' }),
  'ops-terminals-list': makeNode('div'),
  'ops-pty-output': makeNode('pre'),
};

const apiCalls = [];
const delCalls = [];
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

const formatted = ext.formatTerminalList([
  { id: 'pty_1', title: 'Shell', owner: 'user', running: true, exit_code: null },
  { id: 'pty_2', command: 'cargo test', owner: 'model', running: false, exit_code: 0 },
]);
assert.equal(formatted.length, 2);
assert(formatted[0].label.includes('активен'));
assert(formatted[1].label.includes('код 0'));

ext.init({
  $: (id) => els[id] || (els[id] = makeNode('div')),
  api: async (path, body) => {
    apiCalls.push({ path, body });
    if (path === 'sessions/ses_9/terminals' && body) return { id: 'pty_9' };
    if (path === 'sessions/ses_9/terminals') return { terminals: [{ id: 'pty_9', title: 'Shell', owner: 'user', exit_code: null }] };
    if (path.startsWith('sessions/ses_9/terminals/pty_9?')) return { output: 'hi\n', running: true };
    return {};
  },
  del: async (path) => {
    delCalls.push(path);
    return { closed: true };
  },
  node: (tag, text, cls) => Object.assign(makeNode(tag), text !== undefined ? { textContent: String(text) } : {}, cls ? { className: cls } : {}),
  notify: () => {},
  fail: (e) => { throw e; },
  getSessionId: () => 'ses_9',
  getCurrent: () => ({ id: 'ses_9', working_dir: '.' }),
  getProjectId: () => 'default',
  reloadSession: async () => {},
});

(async () => {
  await els['ops-pty-open'].onclick();
  await els['ops-pty-send'].onclick();
  assert.equal(apiCalls[0].path, 'sessions/ses_9/terminals');
  assert.equal(apiCalls[0].body.command, 'echo hi');
  assert(apiCalls.some((c) => c.path === 'sessions/ses_9/terminals/pty_9/input' && c.body.input === 'pwd\n'));
  assert.equal(els['ops-pty-output'].textContent, 'hi\n');
  console.log('PASS Studio PTY terminals formatting and open/read/input wiring');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
