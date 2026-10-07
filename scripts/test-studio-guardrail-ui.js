const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const begin=source.indexOf("$('trace-project').onclick="),end=source.indexOf("$('trace-trash').onclick=",begin);
class Element{constructor(text=''){this.textContent=text;this.children=[];}append(...rows){this.children.push(...rows);}replaceChildren(...rows){this.children=rows;}}
const elements={'trace-status':{value:''},'trace-since':{value:''},'trace-until':{value:''},project:{value:'default'},'trace-guardrail':{value:'blocked'},'trace-guardrail-policy':{value:'a'.repeat(64)},'trace-results':new Element(),'trace-project':{}};
const calls=[],context={URLSearchParams,traceCatalogEpoch:0,$:id=>elements[id],node:(_,text)=>new Element(text),traceView:trace=>new Element(trace.id),fail:error=>{throw error;}};
context.clearTraceBulk=()=>context.traceCatalogEpoch++;
context.api=async path=>{calls.push(path);return {traces:[{id:'trace',project_id:'default'}],total:21};};
vm.createContext(context);vm.runInContext(source.slice(begin,end),context);
(async()=>{
 await elements['trace-project'].onclick();
 assert(calls[0].includes('guardrail=blocked'));assert(calls[0].includes('guardrail_policy_sha256='+'a'.repeat(64)));
 const panel=elements['trace-results'];assert.equal(panel.children[0].textContent,'Найдено вызовов: 21');
 await panel.children.at(-1).children[1].onclick();assert(calls.at(-1).includes('offset=20'));
 const before=calls.length;elements['trace-guardrail-policy'].value='invalid';await elements['trace-project'].onclick();assert.equal(calls.length,before);assert(panel.children[0].textContent.includes('64'));
 elements['trace-guardrail-policy'].value='';let resolve;context.api=()=>new Promise(done=>resolve=done);
 const pending=elements['trace-project'].onclick();elements['trace-guardrail'].value='failed';resolve({traces:[],total:0});await pending;assert(panel.children[0].textContent.includes('64'));
 context.api=async()=>({traces:[{id:'foreign',project_id:'other'}],total:1});await elements['trace-project'].onclick();assert(panel.children[0].textContent.includes('другого проекта'));
 context.api=async path=>{calls.push(path);return {traces:[],total:0};};
 elements['trace-status'].value='failed';elements['trace-since'].value='2026-10-07T12:00';elements['trace-until'].value='2026-10-07T13:00';
 await elements['trace-project'].onclick();const query=new URLSearchParams(calls.at(-1).split('?')[1]);assert.equal(query.get('status'),'failed');assert.equal(query.get('since_ms'),String(new Date('2026-10-07T12:00').getTime()));assert.equal(query.get('until_ms'),String(new Date('2026-10-07T13:00').getTime()));
 const timeBefore=calls.length;elements['trace-since'].value='2026-10-07T14:00';await elements['trace-project'].onclick();assert.equal(calls.length,timeBefore);assert(panel.children[0].textContent.includes('не позже'));
 elements['trace-since'].value='invalid';await elements['trace-project'].onclick();assert.equal(calls.length,timeBefore);assert(panel.children[0].textContent.includes('дата'));
 elements['trace-since'].value='';elements['trace-until'].value='';context.api=()=>new Promise(done=>resolve=done);
 const staleTime=elements['trace-project'].onclick();elements['trace-until'].value='2026-10-07T15:00';resolve({traces:[],total:999});await staleTime;assert(panel.children[0].textContent.includes('дата'));
 console.log('PASS guardrail catalog UI exact filters, pagination, malformed hashes and stale/foreign response rejection, status/time bounds and stale periods');
})().catch(error=>{console.error(error);process.exitCode=1;});
