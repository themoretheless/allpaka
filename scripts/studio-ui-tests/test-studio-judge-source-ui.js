const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const start=source.indexOf('function savedAnswerJudgePanel('),end=source.indexOf('function showEvaluationRun(',start);
assert(start>=0&&end>start);
class Element{constructor(text=''){this.text=text;this.children=[];this.value='';this.open=false;}append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}closest(){return {open:false};}scrollIntoView(){}click(){}}
async function exercise(kind){
 const calls=[],id=new Element(),preview=new Element();let active=true;
 const context={evaluationProject:'default',settings:()=>({provider:'local',model:'mock'}),clearJudgePlan:()=>{},node:(_,text)=>new Element(text),$:name=>name==='judge-plan-id'?id:preview,api:async(path,body)=>{calls.push({path,body});return body?{id:'plan'}:{presets:[{id:'answer_relevance',version:1,name:'Relevance',required_source:'input'}]};}};
 vm.createContext(context);vm.runInContext(source.slice(start,end),context);
 const panel=context.savedAnswerJudgePanel({id:'saved',kind,project_id:'default',dataset_id:'data',dataset_version:2,items:[{sample_id:'a',output:'ready'}]},()=>active);
 panel.open=true;await panel.ontoggle();panel.children[1].children[0].value='answer_relevance';await panel.children[4].onclick();
 assert.equal(calls[1].body[kind==='offline_score'?'offline_score_id':'experiment_id'],'saved');assert.equal(calls[1].body.outputs.a,'ready');assert.equal(id.value,'plan');
 active=false;await panel.children[4].onclick();assert.equal(calls.length,2);
}
(async()=>{await exercise('offline_score');await exercise('experiment');console.log('PASS saved-answer judge UI pins offline/experiment sources and blocks stale creation');})().catch(error=>{console.error(error);process.exitCode=1;});
