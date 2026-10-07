const assert=require('assert'),fs=require('fs'),vm=require('vm');
class Element{constructor(text=''){this.textContent=text;this.children=[];this.hidden=false;}append(...items){for(const item of items)this.children.push(...(item.fragment?item.children:[item]));}replaceChildren(...items){this.children=items;}}
const elements=Object.fromEntries(['sessions','sessions-status','search','project','history-folder','history-sort','history-bookmarked'].map(id=>[id,new Element()]));
for(const element of Object.values(elements)){element.value='';element.checked=false;}
elements['history-folder'].value='active';
let rows=[{id:'a',title:'First',status:'idle'}],error=null,more=[];
const context={$:id=>elements[id],active:'a',URLSearchParams,node:(_,text)=>new Element(text),document:{createDocumentFragment:()=>Object.assign(new Element(),{fragment:true})},api:async()=>{if(error)throw error;return rows;},chatRow:row=>Object.assign(new Element(row.title),{id:row.id}),renderHistoryMore:value=>more.push(value),fail:e=>{throw e;}};
vm.createContext(context);
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
vm.runInContext(source.slice(source.indexOf('let chatListRequest='),source.indexOf('function chatRow(')),context);
(async()=>{
 await context.listChats();const original=elements.sessions.children[0];
 await context.listChats('auto');assert.strictEqual(elements.sessions.children[0],original);
 rows=[{...rows[0],status:'running'}];await context.listChats('auto');assert.notStrictEqual(elements.sessions.children[0],original);
 const changed=elements.sessions.children[0];context.active='b';await context.listChats('auto');assert.notStrictEqual(elements.sessions.children[0],changed);
 const selected=elements.sessions.children[0];elements.search.value='query';await context.listChats('auto');assert.notStrictEqual(elements.sessions.children[0],selected);
 rows=Array.from({length:30},(_,i)=>({id:String(i),title:String(i)}));elements.search.value='';await context.listChats();assert.equal(more.at(-1),true);
 rows=[{id:'next',title:'Next'}];await context.listChats('more');assert.equal(elements.sessions.children.length,31);
 rows=Array.from({length:31},(_,i)=>({id:String(i),title:String(i)}));await context.listChats('auto');const page=elements.sessions.children[30];await context.listChats('auto');assert.strictEqual(elements.sessions.children[30],page);
 error=new Error('offline');await context.listChats('auto');assert.equal(elements.sessions.children.length,0);assert.equal(elements['sessions-status'].hidden,false);
 error=null;await context.listChats('auto');assert.equal(elements.sessions.children.length,31);
 rows=[];await context.listChats();const empty=elements.sessions.children[0];await context.listChats('auto');assert.strictEqual(elements.sessions.children[0],empty);
 console.log('PASS history refresh preserves DOM identity, updates changes, paginates and recovers from errors');
})().catch(error=>{console.error(error);process.exitCode=1;});
