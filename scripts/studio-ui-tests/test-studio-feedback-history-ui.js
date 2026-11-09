const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8'),pending=[],opened=[];
function node(tag,text){return {tag,text,children:[],append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x}};}
const context={node,encodeURIComponent,api:path=>new Promise((resolve,reject)=>pending.push({path,resolve,reject}))};vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function feedbackHistoryPanel('),source.indexOf('function feedbackPanel(')),context);
const receipt=(offset,total=21)=>({kind:'feedback_history',trace_id:'trace-test',order:'version_desc',provider_calls:0,offset,limit:20,total,latest_version:total,has_more:offset+Math.min(20,Math.max(0,total-offset))<total,versions:Array.from({length:Math.min(20,Math.max(0,total-offset))},(_,i)=>({version:total-offset-i,saved_ms:1000,annotation_count:3,active_count:2}))});
(async()=>{
 let current=true;const panel=context.feedbackHistoryPanel('trace-test',()=>current,async(version,active)=>{if(active())opened.push(version)}),load=panel.children[1],body=panel.children[2];
 let task=load.onclick();await load.onclick();assert.equal(pending.length,1);pending[0].resolve(receipt(0));await task;
 assert.equal(body.children.length,23);await body.children[1].children[1].onclick();assert.deepEqual(opened,[21]);
 const oldOpen=body.children[1].children[1],next=body.children[22];task=next.onclick();await oldOpen.onclick();assert.deepEqual(opened,[21]);pending[1].resolve(receipt(20));await task;
 assert.equal(body.children.length,4);await body.children[1].children[1].onclick();assert.deepEqual(opened,[21,1]);assert(body.children[3].disabled);
 task=load.onclick();const before=body.children;current=false;pending[2].resolve(receipt(0));await task;assert.strictEqual(body.children,before);assert(!load.disabled);
 current=true;task=load.onclick();const bad=receipt(0);bad.versions[0].version=9;pending[3].resolve(bad);await task;assert.match(body.children[0].text,/Некорректная/);
 task=load.onclick();current=false;const beforeError=body.children;pending[4].reject(new Error('stale failure'));await task;assert.strictEqual(body.children,beforeError);
 console.log('PASS feedback history pages, selected versions, receipt checks, duplicate exclusion and stale actions/responses/errors');
})().catch(error=>{console.error(error);process.exitCode=1});
