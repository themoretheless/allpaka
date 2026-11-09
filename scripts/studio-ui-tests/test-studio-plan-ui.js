const assert=require('assert'),fs=require('fs'),vm=require('vm');
class Element{
 constructor(tag,text){this.tag=tag;this.textContent=text||'';this.children=[];this.attributes={};this.value='';}
 append(...nodes){for(const node of nodes){node.parent=this;this.children.push(node);}}
 replaceChildren(...nodes){this.children=[];this.append(...nodes);}
 setAttribute(key,value){this.attributes[key]=value;}
 querySelector(selector){for(const node of this.children){if(selector==='input'&&node.tag==='input'||selector==='[data-goal-draft]'&&node.attributes['data-goal-draft'])return node;const found=node.querySelector(selector);if(found)return found;}return null;}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(node=>node!==this);}
}
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8'),elements=Object.fromEntries(['plan','plan-add','plan-revision','plan-checkpoints'].map(id=>[id,new Element('div')])),calls=[];
const items=[{id:'first',title:'First',status:'completed',acceptance:['criterion'],evidence:['reported result']},{id:'next',title:'Next',status:'pending',acceptance:['finish'],evidence:[]}];
const context={active:'session',current:{id:'session',settings:{mode:'goal'},plan_revision:3,plan:items},lastPlan:'',$:id=>elements[id],node:(tag,text)=>new Element(tag,text),api:async(path,body)=>{calls.push({path,body:JSON.parse(JSON.stringify(body))});return {};},poll:async()=>{},notify:()=>{}};
vm.createContext(context);vm.runInContext(source.slice(source.indexOf('let lastPlanCheckpointSignature='),source.indexOf('async function branchAt(')),context);
vm.runInContext(source.slice(source.indexOf("$('plan-add').onclick="),source.indexOf("$('manage-chat').onclick=")),context);
(async()=>{
 context.renderPlan(items);const first=elements.plan.children[0],next=elements.plan.children[1];assert(first.children[3].children[2].disabled);assert(!next.children[3].children[2].disabled);
 const criterion=next.children[3].children[2];criterion.value='Updated criterion\nSecond criterion';criterion.onchange();await Promise.resolve();assert.deepEqual(calls.at(-1).body.steps[1].acceptance,['Updated criterion','Second criterion']);assert.equal(calls.at(-1).body.base_revision,3);assert.equal(calls.at(-1).body.allow_reopen,false);
 first.children[0].onclick();await Promise.resolve();assert.equal(calls.at(-1).body.allow_reopen,true);assert.equal(calls.at(-1).body.steps[0].id,'first');
 const session={id:'session',plan_revision:3,plan_checkpoints:[{revision:2,source:'agent',steps:items},{revision:3,source:'user',steps:items}]};context.renderPlanCheckpoints(session);const checkpointNode=elements['plan-checkpoints'].children[1];context.renderPlanCheckpoints(session);assert.strictEqual(elements['plan-checkpoints'].children[1],checkpointNode);assert(elements['plan-checkpoints'].children[0].textContent.includes('Ранние'));
 session.goal={id:'new-goal'};session.plan_checkpoints[1].goal_id='old-goal';context.renderPlanCheckpoints(session);assert(elements['plan-checkpoints'].children[1].children.some(child=>child.textContent==='Предыдущая цель'));
 session.goal={id:'current-goal'};session.messages=[{role:'user',content:'Original old objective'}];session.plan_checkpoints[1].goal_origin={id:'old-goal',message_index:0};context.renderPlanCheckpoints(session);assert(elements['plan-checkpoints'].children[1].children.some(child=>child.textContent==='Исходная задача: Original old objective'));
 context.renderPlanCheckpoints(null);assert.equal(elements['plan-checkpoints'].children.length,0);assert.equal(elements['plan-revision'].textContent,'');
 elements['plan-add'].onclick();const draft=elements.plan.querySelector('[data-goal-draft]');assert(draft);const before=calls.length;await draft.children[3].onclick();assert.equal(calls.length,before);assert(draft.children[5].textContent.includes('критерий'));
 draft.children[0].value='New milestone';draft.children[2].value='Actual acceptance criterion';await draft.children[3].onclick();assert.equal(calls.at(-1).body.steps.at(-1).title,'New milestone');assert.deepEqual(calls.at(-1).body.steps.at(-1).acceptance,['Actual acceptance criterion']);assert.equal(calls.at(-1).body.base_revision,3);
 context.active='other';const cross=calls.length;await draft.children[3].onclick();assert.equal(calls.length,cross);
 console.log('PASS durable plan UI criteria/reported evidence, explicit reopen, optimistic revision, stable checkpoints and staged Goal milestone creation');
})().catch(error=>{console.error(error);process.exitCode=1;});
