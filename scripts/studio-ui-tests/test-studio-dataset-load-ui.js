const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8'),start=source.indexOf('let selectedDatasetLoadEpoch='),end=source.indexOf("$('evaluation-open').onclick",start);
const pending=[],elements={project:{value:'p'},'evaluation-dialog':{open:true},'evaluation-dataset':{value:'d'},'evaluation-version':{value:2}};
const context={evaluationDataset:null,evaluationProject:'p',encodeURIComponent,$:id=>elements[id],api:()=>new Promise((resolve,reject)=>pending.push({resolve,reject})),showDataset:s=>{context.evaluationDataset=s}};
vm.createContext(context);vm.runInContext(source.slice(start,end),context);
const snapshot=(id='d',version=2)=>({id,version,project_id:'p',sha256:'a'.repeat(64)});
(async()=>{
 const old=context.loadSelectedDataset();elements['evaluation-version'].value=1;const fresh=context.loadSelectedDataset();pending[1].resolve(snapshot('d',1));await fresh;pending[0].resolve(snapshot());await old;assert.equal(context.evaluationDataset.version,1);
 let task=context.loadSelectedDataset();const draft=context.evaluationDataset={id:'d',name:'draft'};pending[2].resolve(snapshot('d',1));await task;assert.strictEqual(context.evaluationDataset,draft);
 task=context.loadSelectedDataset();pending[3].resolve(snapshot('foreign',1));await assert.rejects(task,/несовместимая/);assert.strictEqual(context.evaluationDataset,draft);
 task=context.loadSelectedDataset();elements.project.value='other';pending[4].reject(new Error('old project failure'));await task;assert.strictEqual(context.evaluationDataset,draft);
 elements.project.value='p';task=context.loadSelectedDataset();elements['evaluation-dialog'].open=false;pending[5].resolve(snapshot('d',1));await task;assert.strictEqual(context.evaluationDataset,draft);
 console.log('PASS selected dataset load rejects stale versions, draft replacement, foreign identity, old errors and closed-dialog responses');
})().catch(error=>{console.error(error);process.exitCode=1});
