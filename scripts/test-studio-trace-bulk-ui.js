const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const begin=source.indexOf('let traceBulkEpoch=0;'),end=source.indexOf("$('trace-project').onclick=",begin);
assert(begin>=0&&end>begin);
function setup(){
 const downloads=[],calls=[],project={value:'default'};
 class Element{constructor(tag,text=''){this.tag=tag;this.textContent=text;this.children=[];}append(...rows){this.children.push(...rows);}prepend(...rows){this.children.unshift(...rows);}replaceChildren(...rows){this.children=rows;}click(){if(this.tag==='a')downloads.push(this.download);}}
 const panel=new Element('div'),button=new Element('button');
 const context={Set,Map,JSON,Date,Blob,URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL:()=>{}},setTimeout:fn=>fn(),
  $:id=>({'project':project,'trace-results':panel,'trace-bulk':button}[id]),node:(tag,text)=>new Element(tag,text),traceView:()=>new Element('details')};
 const packet=body=>({kind:'trace_export_batch',schema_version:1,project_id:'default',trace_count:body.trace_ids.length,provider_calls:0,privacy:{feedback_included:body.include_feedback},traces:body.trace_ids.map(id=>({trace:{id,project_id:'default'},feedback:body.include_feedback?{version:1}:null}))});
 context.api=async(path,body)=>{
  calls.push({path,body});if(body)return packet(body);
  const next=path.includes('offset=20');return {total:21,traces:(next?[['last','completed']]:[['first','completed'],['running','running']]).map(([id,status])=>({id,status,project_id:'default',spans:[]}))};
 };
 vm.createContext(context);vm.runInContext(source.slice(begin,end),context);
 return {context,panel,button,project,calls,downloads,packet};
}
function box(row){return row.children[0].children[0];}
(async()=>{
 const good=setup();await good.button.onclick();
 assert.equal(good.panel.children[2].children[0].checked,false);
 assert.equal(box(good.panel.children[7]).disabled,true);
 const first=box(good.panel.children[6]);first.checked=true;first.onchange();
 await good.panel.children.at(-1).children[1].onclick();
 const last=box(good.panel.children[6]);last.checked=true;last.onchange();
 await good.panel.children[3].onclick();
 assert.deepEqual(good.calls.at(-1).body,{project_id:'default',trace_ids:['first','last'],include_feedback:false});assert.equal(good.downloads.length,1);
 good.panel.children[4].onclick();assert.equal(good.panel.children[3].disabled,true);assert.equal(last.checked,false);
 const feedback=setup();await feedback.button.onclick();box(feedback.panel.children[6]).checked=true;box(feedback.panel.children[6]).onchange();feedback.panel.children[2].children[0].checked=true;await feedback.panel.children[3].onclick();assert.equal(feedback.calls.at(-1).body.include_feedback,true);
 for(const mode of ['project','selection','foreign','feedback']){
  const check=setup();await check.button.onclick();box(check.panel.children[6]).checked=true;box(check.panel.children[6]).onchange();
  check.context.api=async(path,body)=>{const packet=check.packet(body);if(mode==='project')check.project.value='other';if(mode==='selection')check.panel.children[4].onclick();if(mode==='foreign')packet.traces[0].trace.project_id='other';if(mode==='feedback')packet.traces[0].feedback={private:'unexpected'};return packet;};
  await check.panel.children[3].onclick();assert.equal(check.downloads.length,0);
 }
 const stale=setup();stale.context.api=async()=>{stale.project.value='other';return {traces:[],total:0};};await stale.button.onclick();assert.equal(stale.panel.children.length,0);
 console.log('PASS bulk trace UI preserves paged selection, explicit feedback, running exclusion and stale/mismatched export rejection');
})().catch(error=>{console.error(error);process.exitCode=1;});
