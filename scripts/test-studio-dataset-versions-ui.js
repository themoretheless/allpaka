const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const start=source.indexOf('let datasetVersionsEpoch='),end=source.indexOf('let datasetCompareEpoch=',start);
const pending=[],requests=[],el=()=>({children:[],append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x}});
const panel=el(),elements={project:{value:'p'},'evaluation-dialog':{open:true},'evaluation-dataset':{value:'d'},'dataset-versions-result':panel,'dataset-versions-show':{},'evaluation-version':{}};
const context={evaluationDataset:{id:'d'},evaluationProject:'p',URLSearchParams,encodeURIComponent,$:id=>elements[id],node:(tag,text)=>Object.assign(el(),{tag,text}),api:path=>{requests.push(path);return new Promise(resolve=>pending.push(resolve))},showDataset:s=>{context.evaluationDataset=s}};
vm.createContext(context);vm.runInContext(source.slice(start,end),context);
const hash='a'.repeat(64),receipt=()=>({id:'d',project_id:'p',provider_calls:0,versions:[{version:2,name:'V2',samples:1,sha256:hash}],offset:0,limit:20,order:'version_desc',total:2,latest_version:2,has_more:true});
(async()=>{
 let task=context.loadDatasetVersions();pending.shift()(receipt());await task;
 const button=panel.children[1].children[2];task=button.onclick();assert(button.disabled);await button.onclick();assert.equal(requests.length,2);
 pending.shift()({id:'d',project_id:'p',version:2,sha256:hash});await task;assert.equal(elements['evaluation-version'].value,2);
 task=context.loadDatasetVersions();context.evaluationDataset={id:'d',name:'new draft'};const before=panel.children;pending.shift()(receipt());await task;assert.strictEqual(panel.children,before);
 task=context.loadDatasetVersions();const bad=receipt();bad.versions[0].sha256='bad';pending.shift()(bad);await task;assert.match(panel.children[0].text,/некорректные/);
 task=context.loadDatasetVersions();pending.shift()(receipt());await task;task=panel.children[1].children[2].onclick();pending.shift()({id:'d',project_id:'p',version:2,sha256:'b'.repeat(64)});await task;assert.match(panel.children[1].children[3].textContent,/отличается/);
 console.log('PASS version history pins snapshot hashes, rejects malformed metadata and duplicate loads, preserves replacement drafts');
})().catch(error=>{console.error(error);process.exitCode=1});
