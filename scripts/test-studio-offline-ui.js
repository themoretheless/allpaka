const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const start=source.indexOf('let offlineScoresEpoch=0,'),end=source.indexOf('let evaluationMatrixEpoch=',start);
assert(start>=0&&end>start);
class Element {constructor(text=''){this.text=text;this.children=[];this.open=false;}append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}}
function setup(){
 const panel=new Element(),button=new Element(),compare=new Element(),comparison=new Element(),baseline={value:'score'},candidate={value:'score'},dialog={open:true},filter={checked:false},dataset={value:'data'},calls=[];
 const score={id:'score',project_id:'default',dataset_id:'data',dataset_version:1};
 const receipt={...score,kind:'offline_score',provider_calls:0,mean_scores:{exact_match:1},items:[{sample_id:'a',output:'<script>untrusted</script>',scores:{exact_match:1}}]};
 const context={evaluationProject:'default',URLSearchParams,metricLabels:{},savedAnswerJudgePanel:()=>new Element('judge-plan'),node:(_,text)=>new Element(text),$:id=>({'offline-scores-result':panel,'offline-scores-refresh':button,'evaluation-dialog':dialog,'offline-scores-compare':compare,'offline-scores-comparison':comparison,'offline-score-baseline':baseline,'offline-score-candidate':candidate,'offline-scores-dataset-only':filter,'evaluation-dataset':dataset}[id]),api:async path=>{calls.push(path);return path.includes('?')?{scores:[score],total:1,invalid_receipts:0,truncated:false,has_more:false}:receipt;}};
 vm.createContext(context);vm.runInContext(source.slice(start,end),context);return {context,panel,button,calls,receipt,compare,comparison,filter,dataset};
}
(async()=>{
 // Model native option semantics: absent value defaults to its visible text.
 const select={children:[],replaceChildren(...items){this.children=items;},get value(){return this.selected ?? this.children[0]?.value ?? '';},set value(value){this.selected=value;}};
 const datasetContext={evaluationProject:'default',api:async()=>({datasets:[{id:'data',name:'Fixture',version:2}]}),$:()=>select,node:(_,text)=>{let explicit;return {dataset:{},get value(){return explicit ?? text;},set value(value){explicit=value;}};}};
 vm.createContext(datasetContext);
 const loaderStart=source.indexOf('async function loadEvaluationDatasets('),loaderEnd=source.indexOf('async function loadSelectedDataset(',loaderStart);
 vm.runInContext(source.slice(loaderStart,loaderEnd),datasetContext);
 await datasetContext.loadEvaluationDatasets();assert.equal(select.value,'');
 assert.equal(select.children[1].value,'data');assert.equal(select.children[1].dataset.version,2);
 await datasetContext.loadEvaluationDatasets('data');assert.equal(select.value,'data');
 const good=setup();await good.button.onclick();const section=good.panel.children[1];section.open=true;await section.ontoggle();
 assert.equal(good.calls.length,2);assert.equal(section.children[1].children[2].children[1].text,'<script>untrusted</script>');
 const paired=setup();paired.context.api=async(path,body)=>{paired.calls.push(path);return body?{kind:'offline_comparison',project_id:'default',baseline_id:'score',candidate_id:'score',regressions:1,improvements:2,pairs:[]} : paired.receipt;};await paired.compare.onclick();assert.equal(paired.calls.length,3);assert(paired.comparison.children[1].text.includes('ухудшения'));
 const blockedPair=setup();blockedPair.receipt.project_id='other';blockedPair.context.api=async path=>{blockedPair.calls.push(path);return blockedPair.receipt;};await blockedPair.compare.onclick();assert.equal(blockedPair.calls.length,2);assert(blockedPair.comparison.children[0].text.includes('проекту'));
 const stalePair=setup();stalePair.context.api=async()=>{stalePair.context.evaluationProject='other';return stalePair.receipt;};await stalePair.compare.onclick();assert.equal(stalePair.comparison.children.length,1);assert(stalePair.comparison.children[0].text.includes('Проверка'));
 const filtered=setup();filtered.filter.checked=true;await filtered.button.onclick();assert(filtered.calls[0].includes('dataset_id=data'));const empty=setup();empty.filter.checked=true;empty.dataset.value='';await empty.button.onclick();assert.equal(empty.calls.length,0);assert(empty.panel.children[0].text.includes('набор'));
 const changing=setup();changing.filter.checked=true;changing.context.api=async()=>{changing.dataset.value='other';return {scores:[]};};await changing.button.onclick();assert(changing.panel.children[0].text.includes('Проверка'));
 const paged=setup();paged.context.api=async path=>{paged.calls.push(path);return {scores:[],total:21,invalid_receipts:0,truncated:false,has_more:!path.includes('offset=20')};};await paged.button.onclick();await paged.panel.children.at(-1).children[0].onclick();assert(paged.calls[1].includes('offset=20'));await paged.panel.children.at(-1).children[0].onclick();assert(paged.calls[2].includes('offset=0'));
 const stale=setup();stale.context.api=async()=>{stale.context.evaluationProject='other';return {scores:[]};};await stale.button.onclick();assert.equal(stale.panel.children.length,1);
 const foreign=setup();foreign.receipt.project_id='other';await foreign.button.onclick();const card=foreign.panel.children[1];card.open=true;await card.ontoggle();assert.equal(card.children[1].children.length,1);assert(card.children[1].children[0].text.includes('выбору'));
 console.log('PASS offline score UI loads verified details, treats outputs as text, and discards stale/foreign receipts');
})().catch(error=>{console.error(error);process.exitCode=1;});
