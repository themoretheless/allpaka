// Exercise the actual import handler with API receipts and a minimal DOM.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const start=source.indexOf('let evaluationMatrixEpoch=0;');
const end=source.indexOf('let studioMatrixDraft=',start);
assert(start>=0&&end>start);
class Element {
  constructor(text=''){this.text=text;this.children=[];this.value='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  scrollIntoView(){}
  setAttribute(){}
}
function setup(report,receipts){
  const input=new Element(),panel=new Element(),dialog={open:true},calls=[];
  input.files=[{size:100,text:async()=>JSON.stringify(report)}];
  const elements={'evaluation-matrix-import':input,'evaluation-matrix-id':new Element(),'evaluation-matrix-open':new Element(),'evaluation-matrices-show':new Element(),'evaluation-matrices-list':new Element(),'evaluation-matrix-result':panel,'evaluation-dialog':dialog};
  const context={evaluationProject:'default',evaluationLabels:{completed:'Готово'},metricLabels:{},TextEncoder,
    $:id=>elements[id],node:(_,text)=>new Element(text),
    api:async(path,body)=>{calls.push({path,body});return body?{regressions:1,improvements:2,pairs:[]} : receipts[path.split('/').pop()];},
    showEvaluationRun:()=>{}};
  vm.createContext(context);vm.runInContext(source.slice(start,end),context);
  return {input,panel,calls,context,elements};
}
const run=id=>({id,project_id:'default',dataset_id:'data',dataset_version:1,dataset_sha256:'hash',
  metrics:['exact_match'],settings:{model:'mock'},status:'completed',strict_quality:true,mean_scores:{exact_match:0.5}});
const report={kind:'client_experiment_matrix',variants:['a','b'].map(id=>({label:id,result:{run:{id},passed:true}}))};
(async()=>{
  const valid=setup(report,{a:run('a'),b:run('b')});await valid.input.onchange();
  assert.equal(valid.calls.length,3);assert.equal(valid.calls[2].body.baseline_id,'a');
  assert(valid.panel.children[2].children.some(node=>node.text.includes('Ухудшений: 1')));
  valid.context.api=async(path,body)=>{valid.calls.push({path,body});return {kind:'experiment_matrix',schema_version:1,id:'matrix',project_id:'default',dataset_id:'data',dataset_version:1,dataset_sha256:'hash',passed:false,variants:[{label:'a',model:'mock',mean_scores:{exact_match:0.5},comparison:{regressions:0,improvements:0}}]};};
  await valid.panel.children.at(-1).onclick();assert.equal(valid.elements['evaluation-matrix-id'].value,'matrix');assert.equal(valid.calls.at(-1).path,'evaluation/matrices');assert.equal(valid.calls.at(-1).body.variants[1].run_id,'b');
  await valid.elements['evaluation-matrix-open'].onclick();assert.equal(valid.calls.at(-1).path,'evaluation/matrices/matrix');assert(valid.panel.children[0].children[1].text.includes('ухудшения'));
  const foreign=setup(report,{a:run('a'),b:{...run('b'),project_id:'foreign'}});await foreign.input.onchange();
  assert.equal(foreign.calls.length,2);assert(foreign.panel.children[0].text.includes('проекте'));
  const duplicate=setup({...report,variants:[report.variants[0],report.variants[0]]},{});await duplicate.input.onchange();
  assert.equal(duplicate.calls.length,0);
  const stale=setup(report,{a:run('a'),b:run('b')});
  stale.context.api=async()=>{stale.context.evaluationProject='changed';return run('a');};
  await stale.input.onchange();assert.equal(stale.panel.children.length,1);
  assert(stale.panel.children[0].text.includes('Проверка'));
  console.log('PASS matrix UI uses authoritative scores/comparisons, rejects foreign/duplicate runs and discards stale imports');
})().catch(error=>{console.error(error);process.exitCode=1;});
