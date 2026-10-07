const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const start=source.indexOf('async function loadTraceSummary('),end=source.indexOf("$('trace-summary-refresh').onclick=",start);
const elements={},pending=[];
const element=id=>elements[id]??=( {value:id==='project'?'project':id==='trace-summary-scope'?'project':id==='trace-summary-status'?'failed':'',open:true,disabled:false,children:[],replaceChildren(...rows){this.children=rows;}} );
const context={URLSearchParams,traceSummaryEpoch:0,traceSummaryProject:'project',traceSummarySession:null,traceSummaryReceipt:null,$:element,firstTextRows:()=>[],evaluationLabels:{},clearTraceSummary:()=>{context.traceSummaryEpoch++;},node:(tag,text)=>({tag,text,children:[],append(...rows){this.children.push(...rows);}}),api:path=>new Promise(resolve=>pending.push({path,resolve}))};
vm.createContext(context);vm.runInContext(source.slice(start,end),context);
const packet={trace_count:0,model_calls:0,input_tokens:0,output_tokens:0,input_tokens_unknown_calls:0,output_tokens_unknown_calls:0,trace_statuses:{},reported_cost_by_currency:{},reported_cost_examples:[],conversations:[],models:[]};
(async()=>{
 const old=context.loadTraceSummary(0),current=context.loadTraceSummary(100);
 assert(pending[1].path.includes('conversation_offset=100')&&pending[1].path.includes('status=failed'));
 pending[0].resolve(packet);await old;
 assert(element('trace-summary-refresh').disabled);
 assert.equal(context.traceSummaryReceipt,null);
 pending[1].resolve(packet);await current;
 assert(!element('trace-summary-refresh').disabled);
 assert(context.traceSummaryReceipt);
 console.log('PASS actual summary page loader retains current busy state and discards stale pages');
})().catch(error=>{console.error(error);process.exitCode=1;});
