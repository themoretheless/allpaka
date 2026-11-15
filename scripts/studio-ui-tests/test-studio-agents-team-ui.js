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
  'ops-team-id': Object.assign(makeNode('input'), { value: 't2' }),
  'ops-team-title': Object.assign(makeNode('input'), { value: 'Verify build' }),
  'ops-team-deps': Object.assign(makeNode('input'), { value: 't1' }),
  'ops-agents-list': makeNode('div'),
  'ops-team-list': makeNode('div'),
};

const apiCalls = [];
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

const board = ext.formatTeamBoard([
  { id: 't1', title: 'Implement', status: 'completed', dependencies: [] },
  { id: 't2', title: 'Test', status: 'pending', dependencies: ['t1'] },
]);
assert.equal(board.length, 2);
assert(board[1].label.includes('← t1'));

ext.init({
  $: (id) => els[id] || (els[id] = makeNode('div')),
  api: async (path, body) => {
    apiCalls.push({ path, body });
    if (path === 'sessions/ses_t/agents') return { agents: [] };
    if (path === 'sessions/ses_t/team' && !body) return { tasks: [] };
    return { ok: true };
  },
  del: async () => ({}),
  node: (tag, text, cls) => Object.assign(makeNode(tag), text !== undefined ? { textContent: String(text) } : {}, cls ? { className: cls } : {}),
  notify: () => {},
  fail: (e) => { throw e; },
  getSessionId: () => 'ses_t',
  getCurrent: () => null,
  getProjectId: () => 'default',
  reloadSession: async () => {},
});

(async () => {
  await els['ops-team-add'].onclick();
  assert.equal(apiCalls[0].path, 'sessions/ses_t/team');
  assert.deepEqual(apiCalls[0].body, {
    action: 'create',
    id: 't2',
    title: 'Verify build',
    dependencies: ['t1'],
  });
  console.log('PASS Studio subagents and Agent Team DAG board formatting and mutations');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
