const assert=require('assert'),fs=require('fs'),vm=require('vm');
const pending=[];function node(tag,text){return {tag,text,children:[],value:'',append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x},setAttribute(){}};}
const context={node,Option:function(text,value){return {text,value}},encodeURIComponent,api:path=>new Promise((resolve,reject)=>pending.push({path,resolve,reject})),notify:()=>{}};
vm.createContext(context);const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');vm.runInContext(source.slice(source.indexOf('function feedbackHistoryPanel('),source.indexOf('function guardrailReceiptRows(')),context);
const receipt=version=>({trace_id:'trace-test',version,annotations:[],summaries:[]});
(async()=>{
 const panel=context.feedbackPanel({id:'trace-test',spans:[]}),content=panel.children[1];panel.open=true;panel.ontoggle();pending[0].resolve(receipt(3));await new Promise(setImmediate);
 const history=content.children[2].children[0];history.value='1';const first=history.onchange();history.value='2';const second=history.onchange();
 pending[2].resolve(receipt(2));await second;assert.match(content.children[0].text,/Версия 2\./);assert(content.children[7].hidden);
 pending[1].resolve(receipt(1));await first;assert.match(content.children[0].text,/Версия 2\./);
 const currentHistory=content.children[2].children[0];currentHistory.value='1';const stale=currentHistory.onchange();const before=content.children;panel.open=false;pending[3].reject(new Error('old error'));await stale;assert.strictEqual(content.children,before);
 panel.open=true;currentHistory.value='3';const wrong=currentHistory.onchange();pending[4].resolve({...receipt(3),trace_id:'trace-other'});await wrong;assert.match(content.children[8].textContent,/не совпадает/);
 const refresh=content.children[1];const request=refresh.onclick();panel.open=false;pending[5].resolve(receipt(4));await request;assert.strictEqual(content.children,before);
 panel.open=true;const latest=content.children[1].onclick();pending[6].resolve(receipt(3));await latest;
 const form=content.children[7];form.children[4].children[0].value='1';const save=form.onsubmit({preventDefault(){}});
 const selected=content.children[2].children[0];selected.value='1';const switchVersion=selected.onchange();pending[8].resolve(receipt(1));await switchVersion;
 pending[7].resolve(receipt(4));await save;assert.match(content.children[0].text,/Версия 1\./);assert(content.children[7].hidden);
 const oldSelector=content.children[2].children[0];oldSelector.value='2';const preclose=oldSelector.onchange();
 panel.open=false;panel.ontoggle();panel.open=true;panel.ontoggle();const reopened=content.children;
 pending[9].resolve(receipt(2));await preclose;assert.strictEqual(content.children,reopened);assert.match(content.children[0].text,/Версия 1\./);
 const filteredPanel=context.feedbackPanel({id:'trace-test',spans:[]});filteredPanel.open=true;filteredPanel.ontoggle();pending[10].resolve({...receipt(2),annotations:[{author:'Alice',metric:'Quality',comment:'Original comment',deleted:false},{author:'Bob',metric:'Quality',comment:'Removed comment',deleted:true}]});await new Promise(setImmediate);
 const list=filteredPanel.children[1].children[6],search=list.children[0],status=list.children[1],count=list.children[2];
 assert.equal(count.textContent,'Показано оценок: 2 из 2');search.value='ALICE';search.oninput();assert(!list.children[3].hidden);assert(list.children[4].hidden);
 search.value='';search.oninput();status.value='deleted';status.onchange();assert(list.children[3].hidden);assert(!list.children[4].hidden);assert.equal(count.textContent,'Показано оценок: 1 из 2');
 const originalForm=filteredPanel.children[1].children[7];search.value='missing';search.oninput();assert.strictEqual(filteredPanel.children[1].children[7],originalForm);assert.equal(count.textContent,'Показано оценок: 0 из 2');
 const scoped=context.feedbackPanel({id:'trace-test',spans:[{id:1,name:'Source'}]},1);scoped.open=true;scoped.ontoggle();pending[11].resolve(receipt(0));await new Promise(setImmediate);assert.equal(scoped.children[1].children[7].children[0].children[0].value,'1');
 console.log('PASS feedback selection rejects out-of-order versions, closed-panel responses/errors and substituted trace IDs');
})().catch(error=>{console.error(error);process.exitCode=1});
