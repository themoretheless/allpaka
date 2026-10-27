const assert=require('assert'),fs=require('fs'),vm=require('vm');
class Element {
 constructor(){this.value='';this.textContent='';this.children=[];this.disabled=false;this.open=false;this.handlers={};this.selectionStart=0;this.selectionEnd=0;}
 replaceChildren(...items){this.children=items;if(items[0]?.value)this.value=items[0].value;}
 append(item){this.children.push(item);}
 addEventListener(name,fn){this.handlers[name]=fn;}
 showModal(){this.open=true;}
 close(){this.open=false;this.handlers.close?.();}
 focus(){}
}
const elements=new Map();const el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const context={window:{},AbortController,setTimeout,clearTimeout,document:{getElementById:el,createElement:()=>new Element(),addEventListener(){},querySelectorAll:()=>[]}};
vm.createContext(context);vm.runInContext(fs.readFileSync('crates/allpaka-chat/web/files.js','utf8'),context);const Files=context.window.StudioFiles;
assert.equal(Files.lines('one\ntwo\nthree\n',2,3),'two\nthree\n');assert.equal(Files.lines('one\ntwo',2,2),'two');assert.equal(Files.lines('one\r\ntwo\r\n',1,1),'one\r\n');assert.throws(()=>Files.lines('one',1,2));assert.throws(()=>Files.lines('one',1.5,2));assert.deepEqual(Array.from(Files.selection('one\ntwo\n',0,4)),[1,1]);assert.deepEqual(Array.from(Files.selection('я\n猫\n',2,3)),[2,2]);assert.equal(Files.selection('x',0,0),null);
let project={id:'p',roots:[{alias:'src'}]},reads=[],attached=[],pending=[];
const panel=Files.create({project:()=>project,attach:f=>attached.push(f),api:(path,body,signal)=>new Promise((resolve,reject)=>pending.push({path,body,signal,resolve,reject}))});
const flush=()=>new Promise(r=>setImmediate(r));
(async()=>{
 el('file-toggle').onclick();assert.equal(pending[0].path,'files/list');assert.equal(pending[0].body.project_id,'p');
 pending.shift().resolve({entries:[{name:'unsafe.html',path:'unsafe.html',directory:false}]});await flush();assert.equal(el('file-list').children.length,1);
 el('file-list').children[0].onclick();const read=pending.shift();assert.equal(read.path,'files/read');
 read.resolve({root:'src',path:'unsafe.html',text:'<script>bad()</script>\ntwo\nthree\n',bytes:40});await flush();assert.equal(el('file-preview').value.startsWith('<script>'),true);assert.equal(el('file-add').disabled,false);
 el('file-line-start').value='2';el('file-line-end').value='3';el('file-add-lines').onclick();assert.equal(attached[0].text,'two\nthree\n');assert.equal(attached[0].name,'src/unsafe.html:L2-L3');assert.equal(attached[0].projectId,'p');
 el('file-add').onclick();assert(attached[1].text.includes('<script>'));
 // A response from a previous project must never become readable/attachable.
 el('file-list').children[0].onclick();const stale=pending.shift();project={id:'q',roots:[{alias:'other'}]};panel.changed();const next=pending.shift();assert(stale.signal.aborted);stale.resolve({root:'src',path:'secret',text:'OLD PROJECT',bytes:11});await flush();assert.equal(el('file-preview').value,'');assert(el('file-add').disabled);
 next.resolve({entries:[],truncated:true});await flush();assert(el('file-status').textContent.includes('часть'));
 // Closing cancels in-flight work and late results cannot repopulate the panel.
 panel.changed();const closing=pending.shift();el('file-panel').close();assert(closing.signal.aborted);closing.resolve({entries:[{name:'late'}]});await flush();assert.equal(el('file-list').children.length,0);
 // Verify the actual composer integration sends selected contents and drops
 // project-scoped attachments after a project change; no provider is called.
 const app=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8'),sent=[];
 const fields={error:{},prompt:{value:'Проверь строки'},project:{value:'p'}};
 const composer={filePanel:null,StudioFiles:{create:opts=>{composer.attach=opts.attach;return {}; }},config:{projects:[]},$:id=>fields[id],api:async(path,body)=>{sent.push({path,body});return {};},project:()=>project,TextEncoder,renderAttachments:()=>{},draftTexts:[],draftImages:[],current:null,layoutFix:false,settings:()=>({project_id:'p',model:'mock'}),active:'chat',pendingRev:0,poll:async()=>{},listChats:async()=>{}};
 vm.createContext(composer);vm.runInContext(app.slice(app.lastIndexOf('filePanel = StudioFiles.create({')),composer);
 composer.attach(attached[0]);assert.throws(()=>composer.attach(attached[0]),/уже добавлен/);
 composer.draftTexts.push({name:'old',text:'DO_NOT_SEND',projectId:'other'});
 vm.runInContext(app.slice(app.indexOf('async function control('),app.indexOf('function modeHelp()',app.indexOf('async function control('))),composer);
 await composer.control('send');const body=sent[0].body;
 assert(body.text.includes('src/unsafe.html:L2-L3'));assert(body.text.includes('two\nthree\n'));assert(!body.text.includes('DO_NOT_SEND'));assert(!body.text.includes('<script>'));assert.equal(composer.draftTexts.length,0);
 console.log('PASS files: line boundaries, literal preview, scoped attachments, cancellation, stale project/close and partial-search status and composer payload');
})().catch(e=>{console.error(e);process.exitCode=1;});
