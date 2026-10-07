// Run the actual UI orchestration against the native mock-provider HTTP fixture.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const [base,serialized]=process.argv.slice(2),candidates=JSON.parse(serialized),calls=[];
const context={TextEncoder,Date,api:async(path,body)=>{
  calls.push(path);const response=await fetch(base+'/api/'+path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-Allpaka-Client':'studio'},...(body?{body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(5000)});
  if(!response.ok)throw new Error('http_'+response.status);return response.json();
}};
vm.createContext(context);const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');vm.runInContext(source.slice(source.indexOf('function verifyStudioMatrixJob('),source.indexOf('let evaluationMatrixEpoch=')),context);
(async()=>{
  const result=await context.runServerStudioMatrix(candidates,{cancelled:()=>false,wait:()=>new Promise(resolve=>setTimeout(resolve,25)),progress:()=>{}});
  assert.equal(result.job.variants.length,3);assert.equal(result.matrix.passed,false);assert.equal(result.matrix.variants[1].comparison.regressions,1);assert.equal(result.matrix.variants[2].comparison.eligible,true);
  assert.equal(calls.filter(path=>path==='evaluation/experiments').length,0);assert.equal(calls.filter(path=>path==='evaluation/matrices').length,0);assert.equal(calls.filter(path=>path==='evaluation/matrix-jobs').length,1);
  assert.deepEqual(await context.api('evaluation/matrices/'+result.matrix.id),result.matrix);
  process.stdout.write(JSON.stringify({id:result.matrix.id,passed:result.matrix.passed}));
})().catch(error=>{console.error(error);process.exitCode=1;});
