const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8'),start=source.indexOf('let evaluationDatasetCatalogEpoch='),end=source.indexOf('let datasetLifecycleEpoch=',start);
const pending=[],select={value:'d',children:[],replaceChildren(...rows){this.children=rows;this.value='';}},elements={project:{value:'p'},'evaluation-dialog':{open:true},'evaluation-dataset':select};
const context={evaluationProject:'p',$:id=>elements[id],node:(tag,text)=>({tag,text,dataset:{}}),api:()=>new Promise(resolve=>pending.push(resolve))};
vm.createContext(context);vm.runInContext(source.slice(start,end),context);
(async()=>{
 const old=context.loadEvaluationDatasets(),fresh=context.loadEvaluationDatasets();
 pending[1]({datasets:[{id:'d',name:'Current',version:1}]});await fresh;assert.equal(select.value,'d');const rows=select.children;
 pending[0]({datasets:[{id:'other',name:'Old',version:1}]});await old;assert.strictEqual(select.children,rows);
 const removed=context.loadEvaluationDatasets();pending[2]({datasets:[]});await removed;assert.equal(select.value,'');
 const cross=context.loadEvaluationDatasets();elements.project.value='other';pending[3]({datasets:[{id:'foreign',name:'Foreign',version:1}]});await cross;assert.equal(select.children.length,1);
 console.log('PASS active dataset picker preserves existing selection, excludes archived choices and ignores stale/cross-project catalog responses');
})().catch(error=>{console.error(error);process.exitCode=1;});
