const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const source=fs.readFileSync('crates/allpaka-chat/web/app.js','utf8');
const handler=source.match(/save.onclick=(async\(\)=>\{save.disabled=true;try\{.*?\}\});actions.append\(save\)/)[1];
(async()=>{for(const value of ['  sk-new-dummy-b332  ','']){const calls=[];const context={save:{},input:{value},p:{id:'deepseek'},api:async(...args)=>calls.push(args),showConnections:async()=>{},notify:()=>{}};await vm.runInNewContext('('+handler+')()',context);const payload=JSON.parse(JSON.stringify(calls[0][1]));assert.deepEqual(payload,value?{key:'sk-new-dummy-b332',persist:true}:{key:'',persist:true,use_current:true});}console.log('PASS: entered replacement wins; empty field preserves current key');})();
