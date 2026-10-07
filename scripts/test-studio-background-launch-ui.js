const assert=require('assert'),fs=require('fs'),vm=require('vm');
class Element{constructor(){this.value='';this.children=[];this.hidden=false;}replaceChildren(...rows){this.children=rows;}}
const ids=['background-launcher','background-name','background-command','background-follow-up','background-timeout','background-start','background-start-result'];const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));const calls=[];
const context={$:id=>elements[id],node:(_,text)=>({textContent:text}),TextEncoder,active:'one',current:{id:'one',folder:'active',settings:{mode:'auto'}},poll:async()=>{},api:async(path,body)=>{calls.push({path,body});return {id:'bg-1'};}};
vm.createContext(context);const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');vm.runInContext(source.slice(source.indexOf('let backgroundLauncherSession='),source.indexOf("let lastBackground = ''")),context);
(async()=>{
context.renderBackgroundLauncher(context.current);assert(!elements['background-launcher'].hidden);
elements['background-name'].value='Сборка';elements['background-command'].value='printf done';elements['background-follow-up'].value='Inspect output.';elements['background-timeout'].value='10';await elements['background-start'].onclick();assert.equal(calls[0].path,'sessions/one/background');assert.equal(calls[0].body.follow_up,'Inspect output.');assert.equal(calls[0].body.name,'Сборка');assert.equal(elements['background-name'].value,'');assert.equal(elements['background-command'].value,'');
elements['background-command'].value='я'.repeat(8001);await elements['background-start'].onclick();assert.equal(calls.length,1);
elements['background-command'].value='printf second';elements['background-follow-up'].value='';let release;context.api=()=>new Promise(resolve=>release=resolve);const pending=elements['background-start'].onclick();context.active='two';context.current={id:'two',folder:'active',settings:{mode:'chat'}};context.renderBackgroundLauncher(context.current);release({id:'old'});await pending;assert(elements['background-launcher'].hidden);assert.equal(elements['background-start-result'].children.length,0);
context.renderBackgroundLauncher(null);assert(elements['background-launcher'].hidden);
console.log('PASS background launcher explicit follow-up, UTF-8 bounds, mode gating and stale chat rejection');
})().catch(error=>{console.error(error);process.exitCode=1;});
