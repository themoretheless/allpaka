const assert=require('assert'),fs=require('fs'),vm=require('vm');
class Element{constructor(){this.children=[];}append(...items){this.children.push(...items);}setAttribute(){}click(){downloads.push(this.download);}}
const calls=[],downloads=[],packets=[];
const context={node:()=>new Element(),api:async path=>{calls.push(path);return {kind:'experiment_export',schema_version:1,metrics:['exact_match'],outputs_included:path.endsWith('true'),items:[]};},URL:{createObjectURL:blob=>{packets.push(blob);return 'blob:fixture';},revokeObjectURL:()=>{}},Blob,setTimeout:fn=>fn(),fail:error=>{throw error;}};
vm.createContext(context);const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
vm.runInContext(source.slice(source.indexOf('function experimentExportCsv('),source.indexOf('function showEvaluationRun(')),context);
(async()=>{const panel=context.experimentExportPanel({id:'run'}),checkbox=panel.children[0].children[0],button=panel.children[1];
assert.equal(checkbox.checked,false);await button.onclick();assert.equal(calls[0],'evaluation/experiments/run/export?include_outputs=false');
checkbox.checked=true;await button.onclick();assert.equal(calls[1],'evaluation/experiments/run/export?include_outputs=true');assert.deepEqual(downloads,['experiment-run.json','experiment-run.json']);assert.equal(button.disabled,false);
assert.equal(JSON.parse(await packets[0].text()).kind,'experiment_export');panel.children[2].value='csv';await button.onclick();assert.equal(downloads[2],'experiment-run.csv');assert((await packets[2].text()).includes('score_exact_match'));assert.strictEqual(context.experimentExportPanel({id:'run'}),panel);assert.equal(checkbox.checked,true);assert.equal(panel.children[2].value,'csv');
const next=context.experimentExportPanel({id:'next'});assert.notStrictEqual(next,panel);assert.equal(next.children[0].children[0].checked,false);assert.equal(next.children[2].value,'json');
let resolve;context.api=()=>new Promise(done=>{resolve=done;});const pending=next.children[1].onclick();assert.equal(next.children[1].disabled,true);await next.children[1].onclick();assert.strictEqual(context.experimentExportPanel({id:'next'}),next);resolve({kind:'experiment_export'});await pending;assert.equal(next.children[1].disabled,false);
console.log('PASS experiment export preferences persist, isolate runs and prevent duplicate pending downloads');})().catch(error=>{console.error(error);process.exitCode=1;});
const specialCsv=context.experimentExportCsv({kind:'experiment_export',schema_version:1,metrics:['exact_match'],outputs_included:true,run_id:'run',dataset_id:'dataset',status:'failed',items:[{sample_id:'=1+1',status:'failed',scores:{},output:'Привет,\n"мир"'}]});
assert(specialCsv.includes('"\'=1+1"'));
assert(specialCsv.includes('"Привет,\n""мир"""'));
assert(specialCsv.includes('"failed"'));

const provenanceCsv=context.experimentExportCsv({kind:'experiment_export',schema_version:1,metrics:[],outputs_included:false,provider:'local',model:'=MODEL',trace_id:'trace',prompt_ref:{id:'prompt',version:3,sha256:'hash'},items:[{sample_id:'input',scores:{}}]});
assert(provenanceCsv.includes('\"provider\",\"model\",\"trace_id\",\"prompt_id\",\"prompt_version\",\"prompt_sha256\"'));
assert(provenanceCsv.includes('\"local\",\"\'=MODEL\",\"trace\",\"prompt\",\"3\",\"hash\"'));
