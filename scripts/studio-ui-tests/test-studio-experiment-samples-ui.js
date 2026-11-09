const assert=require('assert'),fs=require('fs'),vm=require('vm');
class Element{constructor(text){this.textContent=text;this.children=[];this.value='';this.dataset={};}append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}setAttribute(){}}
const context={node:(_,text)=>new Element(text),evaluationLabels:{completed:'Завершено'},metricLabels:{}};vm.createContext(context);
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');vm.runInContext(source.slice(source.indexOf('let experimentSamplesView='),source.indexOf('function showEvaluationRun(')),context);
const items=[{sample_id:'one',status:'completed',output:'Привет мир',scores:{exact_match:1}},
{sample_id:'broken',status:'failed',error:'provider_error',output:'partial',scores:{}},
{sample_id:'waiting',status:'pending',scores:{}},{sample_id:'short',status:'completed',output_truncated:true,output:'<script>literal</script>',scores:{}}];
const panel=context.experimentSamplesPanel({id:'run',items}),[search,status,count,body]=panel.children;
assert.equal(count.textContent,'Показано 4 из 4');body.children[0].open=true;body.children[0].ontoggle();
status.value='problems';status.onchange();assert.equal(body.children.length,2);assert.equal(body.children[0].dataset.sampleId,'broken');
search.value='SCRIPT';search.oninput();assert.equal(body.children.length,1);assert.equal(body.children[0].children[1].textContent,'<script>literal</script>');
search.value='';status.value='all';status.onchange();assert(body.children[0].open);
search.value='ПРИВЕТ';search.oninput();assert.equal(count.textContent,'Показано 1 из 4');
const refreshed=context.experimentSamplesPanel({id:'run',items:[...items,{sample_id:'new',status:'running',scores:{}}]});assert.strictEqual(refreshed,panel);assert.equal(search.value,'ПРИВЕТ');assert.equal(count.textContent,'Показано 1 из 5');
search.value='missing';search.oninput();assert(body.children[0].textContent.includes('Нет примеров'));
const next=context.experimentSamplesPanel({id:'next',items});assert.notStrictEqual(next,panel);assert.equal(next.children[0].value,'');assert.equal(next.children[1].value,'all');
console.log('PASS experiment status/search filters, Unicode, literal output, open state, refresh persistence and run isolation');
const ranked=[{sample_id:'unknown',status:'pending',scores:{}},{sample_id:'high',status:'completed',duration_ms:50,scores:{exact_match:1}},{sample_id:'zero',status:'completed',duration_ms:0,scores:{exact_match:0}},{sample_id:'tie',status:'completed',duration_ms:50,scores:{exact_match:1}}];
const rankedPanel=context.experimentSamplesPanel({id:'ranked',metrics:['exact_match'],items:ranked}),rankedBody=rankedPanel.children[3],sort=rankedPanel.children[4];
sort.value='score:exact_match';sort.onchange();assert.deepEqual(rankedBody.children.map(row=>row.dataset.sampleId),['zero','high','tie','unknown']);
sort.value='slowest';sort.onchange();assert.deepEqual(rankedBody.children.map(row=>row.dataset.sampleId),['high','tie','zero','unknown']);
sort.value='fastest';sort.onchange();assert.equal(rankedBody.children[0].dataset.sampleId,'zero');assert(rankedBody.children[0].children[1].textContent.includes('0.00 с'));
assert.equal(ranked[0].sample_id,'unknown','sorting must not mutate saved receipt');
context.experimentSamplesPanel({id:'ranked',metrics:['exact_match'],items:ranked});assert.equal(sort.value,'fastest');
sort.value='original';sort.onchange();assert.equal(rankedBody.children[0].dataset.sampleId,'unknown');
console.log('PASS score/duration sorting, zero values, unknowns last, stable ties and immutable receipt order');
