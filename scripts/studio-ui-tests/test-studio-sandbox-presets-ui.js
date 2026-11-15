const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const appSource = fs.readFileSync('crates/allpaka-chat/web/app.js', 'utf8');
const settingsFn = appSource.split('\n').find((line) => line.startsWith('function settings()'));
assert(settingsFn, 'Expected single-line function settings() in app.js');

const harnessSrc = fs.readFileSync('crates/allpaka-chat/web/harness-ext.js', 'utf8');

const elements = {};
const stubEl = () => ({ value: '', checked: false });
const ctx = {
  $: (id) => elements[id] || (elements[id] = stubEl()),
  collectSwarm: () => ({ members: [] }),
};
vm.createContext(ctx);
vm.runInContext(settingsFn, ctx);
const ext = vm.runInContext(`${harnessSrc}\n;StudioHarnessExt;`, ctx);

const defaults = ctx.settings();
assert.equal(defaults.sandbox_mode, 'unsandboxed');
assert.equal(defaults.approval_policy, 'never');
assert.equal(defaults.permission_preset, null);
assert.equal(defaults.team_mode, false);

const controls = {
  presetEl: ctx.$('permission-preset'),
  sandboxEl: ctx.$('sandbox-mode'),
  approvalEl: ctx.$('approval-policy'),
  writesEl: ctx.$('writes'),
  modeEl: ctx.$('mode'),
  stepsEl: ctx.$('steps'),
  teamEl: ctx.$('team-mode'),
};

const expectedPresets = {
  readonly_audit: { sandbox_mode: 'read_only', approval_policy: 'commands', allow_writes: false },
  workspace_sandboxed: { sandbox_mode: 'workspace_write', approval_policy: 'never', allow_writes: true },
  interactive_guarded: { sandbox_mode: 'workspace_write', approval_policy: 'writes_and_commands', allow_writes: true },
  full_trusted: { sandbox_mode: 'unsandboxed', approval_policy: 'never', allow_writes: true },
};

for (const [presetId, expected] of Object.entries(expectedPresets)) {
  const applied = ext.applyPresetFields(presetId, controls);
  assert.deepEqual(applied, expected);
  const s = ctx.settings();
  assert.equal(s.permission_preset, presetId);
  assert.equal(s.sandbox_mode, expected.sandbox_mode);
  assert.equal(s.approval_policy, expected.approval_policy);
  assert.equal(s.allow_writes, expected.allow_writes);
}

const profileApplied = ext.applyProfileFields(
  {
    id: 'coding-safe',
    name: 'Coding Safe',
    mode: 'goal',
    sandbox_mode: 'workspace_write',
    approval_policy: 'writes_and_commands',
    allow_writes: true,
    max_steps: 20,
    team_mode: true,
  },
  controls,
);
assert.equal(profileApplied.permission_preset, 'custom');
const afterProfile = ctx.settings();
assert.equal(afterProfile.permission_preset, 'custom');
assert.notEqual(afterProfile.permission_preset, 'coding-safe');
assert.equal(afterProfile.sandbox_mode, 'workspace_write');
assert.equal(afterProfile.approval_policy, 'writes_and_commands');
assert.equal(afterProfile.team_mode, true);

console.log('PASS Studio sandbox presets, profile customization and settings serialization');
