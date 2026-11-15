const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('crates/allpaka-chat/web/index.html', 'utf8');
const harnessSrc = fs.readFileSync('crates/allpaka-chat/web/harness-ext.js', 'utf8');

// Strict CSP checks: no inline scripts, inline event handlers, or inline style attributes
assert(html.includes('<script src="/harness-ext.js" defer></script>'), 'Missing deferred /harness-ext.js tag');
assert(!/<script(?![^>]*\bsrc=)[^>]*>/i.test(html), 'Found forbidden inline <script> in index.html');
assert(!/\son[a-z]+\s*=/i.test(html), 'Found forbidden inline event handler attribute in index.html');
assert(!/\sstyle\s*=/i.test(html), 'Found forbidden inline style= attribute in index.html');
assert(!/\sstyle\s*=\s*['"]/i.test(harnessSrc), 'Found inline style attribute string in harness-ext.js');

for (const id of ['ops-open', 'ops-dialog', 'interaction-banner', 'session-cwd-badge', 'permission-preset', 'sandbox-mode', 'approval-policy', 'team-mode']) {
  assert(html.includes(`id="${id}"`), `Missing #${id} in index.html`);
}

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

const els = {
  'ops-bundle-json': makeNode('textarea'),
  'ops-automation-output': makeNode('pre'),
  'ops-share-label': Object.assign(makeNode('input'), { value: 'Audit link' }),
  'ops-shares-list': makeNode('div'),
  'ops-profiles-list': makeNode('div'),
  'ops-schedules-list': makeNode('div'),
  'ops-webhooks-list': makeNode('div'),
  'ops-wh-name': Object.assign(makeNode('input'), { value: 'CI Hook' }),
  'ops-wh-template': Object.assign(makeNode('input'), { value: 'Handle {{payload}}' }),
  'ops-wf-name': Object.assign(makeNode('input'), { value: 'Review' }),
  'ops-wf-stages': Object.assign(makeNode('textarea'), { value: '[{"id":"s1","name":"S1","prompt":"check"}]' }),
  'ops-workflows-list': makeNode('div'),
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

ext.init({
  $: (id) => els[id] || (els[id] = makeNode('div')),
  api: async (path, body) => {
    apiCalls.push({ path, body });
    if (path === 'bundles/export?project_id=proj1') return { version: 1, project_id: 'proj1', workflows: [] };
    if (path === 'webhooks' && body) return { webhook: { id: 'whk_1', name: 'CI Hook' }, secret: 'whsec_once' };
    return { workflows: [], profiles: [], schedules: [], webhooks: [], shares: [] };
  },
  del: async () => ({}),
  node: (tag, text, cls) => Object.assign(makeNode(tag), text !== undefined ? { textContent: String(text) } : {}, cls ? { className: cls } : {}),
  notify: () => {},
  fail: (e) => { throw e; },
  getSessionId: () => 'ses_a',
  getCurrent: () => null,
  getProjectId: () => 'proj1',
  reloadSession: async () => {},
});

(async () => {
  await els['ops-bundle-export'].onclick();
  assert(apiCalls.some((c) => c.path === 'bundles/export?project_id=proj1' && c.body === undefined));
  assert(els['ops-bundle-json'].value.includes('"project_id": "proj1"'));

  await els['ops-wh-create'].onclick();
  assert(els['ops-automation-output'].textContent.includes('whsec_once'));

  await els['ops-share-create'].onclick();
  assert(apiCalls.some((c) => c.path === 'sessions/ses_a/shares' && c.body && c.body.label === 'Audit link'));

  await els['ops-wf-save'].onclick();
  assert(apiCalls.some((c) => c.path === 'workflows' && c.body && c.body.name === 'Review'));

  console.log('PASS Studio CSP cleanliness and Operations workflow/bundle/webhook/share wiring');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
