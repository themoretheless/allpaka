const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const candidates=()=>['base','next'].map(label=>({label,sha256:'hash',request:{dataset_id:'data',dataset_version:1,metrics:['exact_match'],settings:{model:label,project_id:'project',mode:'chat',allow_writes:false},prompt_template:'{{input}}'}}));
function setup(){
  const calls=[];let stopped=false;
  const job={kind:'experiment_matrix_job',schema_version:1,id:'job',project_id:'project',status:'running',matrix_id:null,automatic_replay:false,automatic_promotion:false,variants:[{label:'base',run_id:null},{label:'next',run_id:null}]};
  const matrix={kind:'experiment_matrix',schema_version:1,id:'matrix',project_id:'project',dataset_sha256:'hash',variants:[{label:'base',run_id:'run-1'},{label:'next',run_id:'run-2'}]};
  const context={TextEncoder,Date,api:async(path,body)=>{
    calls.push({path,body:body?JSON.parse(JSON.stringify(body)):undefined});
    if(path==='evaluation/matrix-jobs')return JSON.parse(JSON.stringify(job));
    if(path.endsWith('/cancel')){stopped=true;return {cancel_requested:true};}
    if(path==='evaluation/matrix-jobs/job')return {...job,status:stopped?'cancelled':'completed',matrix_id:stopped?null:'matrix',variants:[{label:'base',run_id:'run-1'},{label:'next',run_id:'run-2'}]};
    if(path==='evaluation/matrices/matrix')return matrix;
    throw new Error('unexpected request '+path);
  }};
  vm.createContext(context);vm.runInContext(source.slice(source.indexOf('function verifyStudioMatrixJob('),source.indexOf('let evaluationMatrixEpoch=')),context);return {context,calls,job,matrix};
}
const hooks=()=>({cancelled:()=>false,wait:async()=>{},progress:()=>{}});
(async()=>{
  const good=setup(),draft=candidates(),options=hooks();draft[1].promptSha256='pinned';
  options.progress=()=>{draft[1].request.settings.model='mutated';};
  const result=await good.context.runServerStudioMatrix(draft,options);
  assert.equal(result.matrix.id,'matrix');assert.equal(result.job.id,'job');assert.equal(good.calls[0].body.variants[1].request.settings.model,'next');assert.equal(good.calls[0].body.variants[1].prompt_sha256,'pinned');
  assert.equal(good.calls.filter(call=>call.path==='evaluation/matrix-jobs').length,1);assert(!good.calls.some(call=>call.path==='evaluation/experiments'||call.path==='evaluation/matrices'),'UI never sequences inference or saves a duplicate receipt');
  const invalid=setup(),wrong=candidates();wrong[1].request.settings.project_id='other';await assert.rejects(invalid.context.runServerStudioMatrix(wrong,hooks()));assert.equal(invalid.calls.length,0);
  const stopped=setup(),stopHooks=hooks();let cancel=false;stopHooks.cancelled=()=>cancel;stopHooks.progress=()=>cancel=true;
  try{await stopped.context.runServerStudioMatrix(candidates(),stopHooks);assert.fail();}catch(error){assert.equal(error.matrixJobId,'job');assert.equal(error.matrixJob.status,'cancelled');}
  assert.equal(stopped.calls.filter(call=>call.path.endsWith('/cancel')).length,1);
  const detached=setup(),detachedHooks=hooks();detachedHooks.detached=()=>true;
  try{await detached.context.runServerStudioMatrix(candidates(),detachedHooks);assert.fail();}catch(error){assert.equal(error.matrixJobId,'job');}
  assert.equal(detached.calls.length,1,'detaching observation leaves the server job running');
  const offline=setup(),api=offline.context.api;offline.context.api=async(path,body)=>{if(path==='evaluation/matrix-jobs/job')throw new Error('offline');return api(path,body);};
  try{await offline.context.runServerStudioMatrix(candidates(),hooks());assert.fail();}catch(error){assert.equal(error.matrixJobId,'job');}
  assert.equal(offline.calls.length,1,'network loss never cancels or restarts a server job');
  const substituted=setup();substituted.matrix.variants[1].run_id='foreign';await assert.rejects(substituted.context.runServerStudioMatrix(candidates(),hooks()));
  const foreign=setup();foreign.job.project_id='other';await assert.rejects(foreign.context.runServerStudioMatrix(candidates(),hooks()));assert.equal(foreign.calls.length,1);
  console.log('PASS server matrix UI frozen requests, single admission, explicit stop, detach/network ownership and verified result identity');
})().catch(error=>{console.error(error);process.exitCode=1;});
