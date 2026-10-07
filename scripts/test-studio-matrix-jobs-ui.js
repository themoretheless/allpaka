const assert=require('assert'),fs=require('fs'),vm=require('vm');
class Element{constructor(text){this.textContent=text||'';this.value='';this.children=[];this.disabled=false;}append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}click(){return this.onclick?.();}}
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
function setup(){
  const elements=Object.fromEntries(['matrix-job-open','matrix-job-id','matrix-jobs-show','matrix-jobs-list','matrix-job-result','evaluation-matrix-id'].map(id=>[id,new Element()]));
  const calls=[],job={kind:'experiment_matrix_job',schema_version:1,id:'job',project_id:'project',status:'interrupted',error:'process_restart',matrix_id:null,automatic_replay:false,automatic_promotion:false,variants:[{label:'base',run_id:'run-1'},{label:'next',run_id:null}]};
  const matrix={id:'matrix',kind:'experiment_matrix',schema_version:1,project_id:'project',variants:[{label:'base',run_id:'run-1'},{label:'next',run_id:'run-2'}]};
  const context={TextEncoder,evaluationProject:'project',$:id=>elements[id],node:(_,text)=>new Element(text),savedExperimentMatrixPanel:receipt=>new Element('verified '+receipt.id),api:async(path,body)=>{
    calls.push({path,body});
    if(path.startsWith('evaluation/matrix-jobs?')){const offset=Number(new URLSearchParams(path.split('?')[1]).get('offset'));return {jobs:[JSON.parse(JSON.stringify(job))],offset,has_more:offset===0};}
    if(path.endsWith('/resume')){job.status='running';job.error=null;return {};}
    if(path.endsWith('/retry'))return {...job,id:'retry-job',status:'running',retry_of:job.id,previous_run_ids:job.variants.map(row=>row.run_id)};
    if(path.endsWith('/cancel')){job.status='cancelled';return {cancel_requested:true};}
    if(path==='evaluation/matrix-jobs/job')return JSON.parse(JSON.stringify(job));
    if(path==='evaluation/matrices/matrix')return JSON.parse(JSON.stringify(matrix));
    throw new Error('unexpected '+path);
  }};
  vm.createContext(context);vm.runInContext(source.slice(source.indexOf('function verifyStudioMatrixJob('),source.indexOf('let evaluationMatrixEpoch=')),context);
  vm.runInContext(source.slice(source.indexOf('let studioMatrixJobViewEpoch='),source.indexOf('let evaluationPrompt=')),context);
  elements['matrix-job-id'].value='job';return {context,elements,calls,job,matrix};
}
const button=(panel,text)=>panel.children.find(row=>row.textContent===text);
(async()=>{
  const good=setup();await good.elements['matrix-job-open'].click();let panel=good.elements['matrix-job-result'].children[0];
  const oldResume=button(panel,'Продолжить готовые и неотправленные варианты');assert(oldResume);await oldResume.click();assert(good.calls.some(call=>call.path==='evaluation/matrix-jobs/job/resume'));
  panel=good.elements['matrix-job-result'].children[0];await button(panel,'Остановить задание').click();assert.equal(good.job.status,'cancelled');
  const before=good.calls.length;await oldResume.click();assert.equal(good.calls.length,before,'replaced panel controls cannot replay actions');
  good.job.status='completed';good.job.matrix_id='matrix';good.job.variants=JSON.parse(JSON.stringify(good.matrix.variants));await good.elements['matrix-job-open'].click();
  panel=good.elements['matrix-job-result'].children[0];const open=button(panel,'Открыть готовую матрицу');await open.click();assert.equal(good.elements['evaluation-matrix-id'].value,'matrix');assert(panel.children.some(row=>row.textContent==='verified matrix'));
  good.context.evaluationProject='other';const cross=good.calls.length;await open.click();assert.equal(good.calls.length,cross);
  good.context.evaluationProject='project';good.matrix.variants[1].run_id='foreign';await open.click();assert(panel.children.some(row=>row.textContent.includes('не соответствует')));
  const catalog=setup();await catalog.elements['matrix-jobs-show'].click();assert(button(catalog.elements['matrix-jobs-list'],'Следующие задания'));await button(catalog.elements['matrix-jobs-list'],'Следующие задания').click();assert(catalog.calls.at(-1).path.includes('offset=30'));assert(button(catalog.elements['matrix-jobs-list'],'Предыдущие задания'));
  const stale=setup();let resolve;stale.context.api=()=>new Promise(done=>resolve=done);const pending=stale.elements['matrix-job-open'].click();stale.context.evaluationProject='other';resolve(stale.job);await pending;assert.equal(stale.elements['matrix-job-result'].children.length,0);
  const retrying=setup();await retrying.elements['matrix-job-open'].click();const retryPanel=retrying.elements['matrix-job-result'].children[0],retryButton=button(retryPanel,'Создать новое задание и повторить неудачные варианты');assert(retryButton);assert(retryPanel.children.some(row=>row.textContent.includes('новые вызовы моделей')));
  await retryButton.click();assert.equal(retrying.elements['matrix-job-id'].value,'retry-job');assert.equal(retrying.job.status,'interrupted');assert(retrying.elements['matrix-job-result'].children[0].children.some(row=>row.textContent==='Повтор задания: job'));
  const retryCount=retrying.calls.length;await retryButton.click();assert.equal(retrying.calls.length,retryCount,'old retry action cannot submit twice');
  await button(retrying.elements['matrix-job-result'].children[0],'Открыть исходное задание').click();assert.equal(retrying.elements['matrix-job-id'].value,'job');assert(retrying.elements['matrix-job-result'].children[0].children[0].textContent.includes('interrupted'));
  const forged=setup();await forged.elements['matrix-job-open'].click();forged.context.api=async()=>({...forged.job,id:'retry-job',retry_of:'other',previous_run_ids:forged.job.variants.map(row=>row.run_id)});await button(forged.elements['matrix-job-result'].children[0],'Создать новое задание и повторить неудачные варианты').click();assert.equal(forged.elements['matrix-job-id'].value,'job');
  console.log('PASS matrix job UI open, resume/stop/retry lineage, parent navigation, verified matrix, catalog pages and project isolation');
})().catch(error=>{console.error(error);process.exitCode=1;});
