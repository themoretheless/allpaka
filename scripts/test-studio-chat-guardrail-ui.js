const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const fn=source.split('\n').find(line=>line.startsWith('function settings()'));
const elements={};const context={$:id=>elements[id]||(elements[id]={value:'',checked:false}),collectSwarm:()=>({})};vm.createContext(context);vm.runInContext(fn,context);
assert.equal(context.settings().guardrails,null);
for(const id of ['chat-guardrails-input','chat-guardrails-output','chat-guardrails-action'])context.$(id);elements['chat-guardrails-enabled'].checked=true;elements['chat-guardrails-input'].value=' '+ 'a'.repeat(64)+' ';elements['chat-guardrails-output'].value='b'.repeat(64);elements['chat-guardrails-action'].value='block';
assert.equal(context.settings().guardrails.input_policy_sha256,'a'.repeat(64));assert.equal(context.settings().guardrails.output_policy_sha256,'b'.repeat(64));assert.equal(context.settings().guardrails.action,'block');elements['chat-guardrails-enabled'].checked=false;assert.equal(context.settings().guardrails,null);console.log('PASS Chat guardrails explicit enable, pinned pair/action and disabling');
