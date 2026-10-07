const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const begin=source.indexOf('function guardrailReceiptRows('),end=source.indexOf('let traceSummaryEpoch=',begin);
class Element{constructor(text=''){this.text=text;this.textContent=text;this.children=[];this.style={};}append(...rows){this.children.push(...rows);}prepend(...rows){this.children.unshift(...rows);}replaceChildren(...rows){this.children=rows;}}
const calls=[],project={value:'default'};let feedbackCalls=0;
const context={node:(_,text)=>new Element(text),evaluationLabels:{},reviewQueueCreationPanel:()=>new Element('queue'),feedbackPanel:()=>{feedbackCalls++;return new Element('feedback');},
  $:()=>project,api:async(path)=>{calls.push(path);return {};},clearTraceSummary:()=>{},fail:error=>{throw error;}};
vm.createContext(context);vm.runInContext(source.slice(begin,end),context);
const trace={id:'trace-fixture',project_id:'default',started_ms:100,status:'completed',spans:[{id:0,parent_id:null,kind:'model',name:'m',status:'completed',duration_ms:1,usage:{}}]};
(async()=>{
 const active=context.traceView({...trace,status:'running'});assert(!active.children.some(row=>row.text==='В корзину'));
 const view=context.traceView(trace);await view.children.find(row=>row.text==='В корзину').onclick();
 assert.equal(calls.pop(),'observability/traces/trace-fixture/remove');
 assert.equal(view.children[0].text,'Вызов перемещён в корзину');
 await view.children.find(row=>row.text==='Восстановить вызов').onclick();
 assert.equal(calls.pop(),'observability/traces/trace-fixture/restore');
 const before=feedbackCalls,trash=context.traceView(trace,0,true);assert.equal(feedbackCalls,before);
 await trash.children.find(row=>row.text==='Восстановить вызов').onclick();assert.equal(calls.pop(),'observability/traces/trace-fixture/restore');
 const stale=context.traceView(trace);project.value='changed';await stale.children.find(row=>row.text==='В корзину').onclick();assert.equal(calls.length,0);
 assert(stale.children.some(row=>row.textContent?.includes('Проект изменился')));
 console.log('PASS trace trash UI removal/restore/undo, active exclusion, hidden feedback and stale-project protection');
})().catch(error=>{console.error(error);process.exitCode=1;});
