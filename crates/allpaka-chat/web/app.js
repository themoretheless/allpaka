'use strict';
const $ = id => document.getElementById(id);
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === 'dark';
  const label = dark ? 'Включить светлую тему' : 'Включить тёмную тему';
  $('theme-toggle').textContent = dark ? '☀' : '☾';
  $('theme-toggle').title = label;
  $('theme-toggle').setAttribute('aria-label', label);
}
let savedTheme = 'dark';
try { if (localStorage.getItem('allpaka.studio.theme') === 'light') savedTheme = 'light'; } catch {}
applyTheme(savedTheme);
$('theme-toggle').onclick = () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(theme);
  try { localStorage.setItem('allpaka.studio.theme', theme); } catch {}
};
let config, active = null, current = null, draftImages = [], draftTexts = [], editingProject = '', lastMessages = '', polling = false;
const catalogs = new Map();
const welcome = $('messages').innerHTML;
const stateNames = {idle:'Готов',running:'В работе',paused:'Приостановлен',error:'Требует внимания'};
async function api(path, body) {
  const r = await fetch('/api/' + path, {method:body === undefined?'GET':'POST',headers:body === undefined?{}:{'Content-Type':'application/json','X-Allpaka-Client':'studio'},body:body === undefined?undefined:JSON.stringify(body)});
  const value = await r.json(); if (!r.ok) throw new Error(value.error || `HTTP ${r.status}`); return value;
}
function notify(text) {$('toast').textContent=text;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,4000);}
function fail(e) {$('error').textContent=e.message || String(e);$('error').hidden=false;}
function node(tag,text,className) {const n=document.createElement(tag);if(text !== undefined)n.textContent=text;if(className)n.className=className;return n;}
function settings() {return {project_id:$('project').value,provider:$('provider').value,model:$('model').value.trim(),mode:$('mode').value,max_steps:Number($('steps').value),max_output_tokens:Number($('output-tokens').value),auto_compact:$('auto-compact').checked,compact_threshold:Number($('compact-threshold').value),allow_writes:$('writes').checked};}
async function loadConfig() {
  const selected=$('project').value, selectedProvider=$('provider').value;
  config=await api('config');
  $('project').replaceChildren(...config.projects.map(p=>{const o=node('option',p.name);o.value=p.id;return o;}));
  if(config.projects.some(p=>p.id===selected))$('project').value=selected;
  $('provider').replaceChildren(...config.providers.map(p=>{const o=node('option',p.name+(p.configured?'':' · ключ не задан'));o.value=p.id;return o;}));
  if(selectedProvider)$('provider').value=selectedProvider;
  renderContext();
}
function renderContext() {
  const p=config.projects.find(p=>p.id===$('project').value);if(!p)return;
  $('project-name').textContent=p.name;
  $('roots').replaceChildren(...p.roots.map(r=>{const div=node('div',undefined,'root-card');div.append(node('b',(r.repository?'⑂ ':'▱ ')+r.alias),node('small',r.path),node('small',r.writable?'Auto: запись разрешена для папки':'Только чтение'));return div;}));
}
let chatListRequest=0;
async function listChats() {
  const version=++chatListRequest;
  const params=new URLSearchParams({q:$('search').value,project:$('project').value,folder:$('history-folder').value});
  const rows=await api('sessions?'+params);
  if(version!==chatListRequest)return;
  $('sessions').replaceChildren(...rows.map(r=>{
    const b=node('button',undefined,r.id===active?'active':'');b.append(node('span',r.title),node('small',`${r.provider} · ${stateNames[r.status]||r.status}`));if(r.match_preview)b.append(node('small',r.match_preview));b.onclick=()=>openChat(r.id).catch(fail);return b;
  }));
}
function resetChat() {$('compact-summary').textContent='Контекст ещё не сжат.';$('branch-origin').hidden=true;active=null;current=null;lastMessages='';$('messages').innerHTML=welcome;$('title').textContent='Новый разговор';$('error').hidden=true;$('notice').hidden=true;$('queue').replaceChildren();$('plan').replaceChildren(node('li','План появится во время работы','muted'));renderStatus('idle');listChats().catch(fail);bindSuggestions();}
function bindSuggestions() {document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;$('mode').value=b.dataset.mode;modeHelp();$('prompt').focus();});}
function renderStatus(status) {
  $('manage-chat').disabled=!active;const hidden=current&&current.folder&&current.folder!=='active';$('send').disabled=!!hidden;$('resume').disabled=!!hidden;
  const running=status==='running';$('compact').disabled=!active||running||!!hidden;$('status').textContent=stateNames[status]||status;$('status').className='badge '+status;
  $('send').textContent=running?'В очередь ↑':'Отправить ↑';$('steer').hidden=!running;$('send-now').hidden=!running;$('stop').hidden=!running;$('resume').hidden=!['paused','error'].includes(status);
}
async function openChat(id) {
  active=id;lastMessages='';const s=await api('sessions/'+id);if(active!==id)return;
  $('auto-compact').checked=s.settings.auto_compact??true;$('compact-threshold').value=s.settings.compact_threshold??24000;current=s;$('history-folder').value=s.folder||'active';$('project').value=s.settings.project_id;$('provider').value=s.settings.provider;$('model').value=s.settings.model;$('mode').value=s.settings.mode;$('steps').value=s.settings.max_steps;$('output-tokens').value=s.settings.max_output_tokens??8192;$('writes').checked=s.settings.allow_writes;
  renderModelInfo();renderContext();modeHelp();render(s);await listChats();
}
function render(s) {
  $('compact').disabled=s.status==='running'||s.folder&&s.folder!=='active';$('compact-summary').textContent=s.compaction?.summary||'Контекст ещё не сжат.';
  current=s;$('title').textContent=s.title;renderStatus(s.status);
  $('branch-origin').hidden=!s.parent;
  if(s.parent){const origin=node('button','← Исходный разговор');origin.onclick=()=>openChat(s.parent.session_id).catch(fail);$('branch-origin').replaceChildren(origin,node('span','Независимая ветка · файлы проекта общие'));}
  if(s.error){$('error').textContent=s.error;$('error').hidden=false;}else{$('error').hidden=true;}
  const notice=s.folder&&s.folder!=='active'?'Разговор в архиве или корзине. Откройте «Разговор» → «Восстановить», чтобы продолжить.':s.notice;
  $('notice').textContent=notice||'';$('notice').hidden=!notice;
  const serialized=JSON.stringify([s.messages,s.status]);
  if(serialized!==lastMessages){
    const pane=$('messages'),atBottom=pane.scrollHeight-pane.scrollTop-pane.clientHeight<120;
    pane.replaceChildren(...s.messages.map((m,index)=>{
      const div=node('article',undefined,'message '+m.role);div.append(node('div',m.role==='user'?'ВЫ':m.role==='tool'?'ИНСТРУМЕНТ':'ALLPAKA','role'));
      if(m.role==='tool'){const details=node('details');details.append(node('summary','Результат инструмента'));let result;try{result=JSON.parse(m.content);}catch{}if(typeof result?.diff==='string'){const {diff,...info}=result;details.append(node('pre',JSON.stringify(info,null,2)));if(diff)details.append(StudioContent.render('```diff\n'+diff+'\n```'));}else details.append(StudioContent.render(m.content));div.append(details);}else{div.append(StudioContent.render(m.content||((m.tool_calls||[]).length?'Вызовы инструментов':m.truncated?'Лимит достигнут до текстового ответа. Увеличьте лимит ответа и продолжите.':s.status==='running'?'…':'Ответ не получен.')));}
      if(m.truncated)div.append(node('div','Неполный ответ · достигнут лимит токенов','muted'));
      for(const call of m.incomplete_tool_calls||[]){const d=node('details');d.append(node('summary','Незавершённый вызов · не выполнен'),node('pre',JSON.stringify(call,null,2)));div.append(d);}
      for(const i of m.images||[]){const img=document.createElement('img');img.alt=i.name;img.src=`data:${i.mime};base64,${i.data}`;div.append(img);}
      for(const call of m.tool_calls||[]){const d=node('details');d.append(node('summary',call.function?.name||'Вызов инструмента'),node('pre',call.function?.arguments||''));div.append(d);}
      if(s.status!=='running') {
        const actions=node('div',undefined,'message-actions');
        if(m.role==='assistant'&&!m.truncated&&!(m.tool_calls||[]).length) {
          const branch=node('button','Продолжить в новой ветке');branch.onclick=()=>branchAt(index+1).catch(fail);actions.append(branch);
        }
        if(m.role==='user'&&(index===0||(s.messages[index-1].role==='assistant'&&!s.messages[index-1].truncated&&!(s.messages[index-1].tool_calls||[]).length))) {
          const edit=node('button','Изменить в новой ветке');edit.onclick=()=>branchAt(index,m).catch(fail);actions.append(edit);
          const retry=node('button','Повторить в новой ветке');retry.onclick=()=>branchAt(index,m,true).catch(fail);actions.append(retry);
        }
        if(actions.childNodes.length)div.append(actions);
      }
      return div;
    }));lastMessages=serialized;if(atBottom)pane.scrollTop=pane.scrollHeight;
  }
  $('queue').replaceChildren(...s.queue.map((p,i)=>node('div',`${i+1}. ${p.text.slice(0,100)}`,'chip')),...s.steering.map(t=>node('div','Steer: '+t.slice(0,100),'chip')));
  if(s.queue.length||s.steering.length){const b=node('button','Очистить очередь');b.onclick=()=>control('clear_queue').catch(fail);$('queue').append(b);}
  $('plan').replaceChildren(...(s.plan.length?s.plan.map(p=>node('li',p.title,p.status)):[node('li','План появится во время работы','muted')]));
  const u=s.usage||{};const input=u.prompt_tokens??u.input_tokens;const output=u.completion_tokens??u.output_tokens;
  $('usage').textContent=`Шаг ${s.step} / ${s.settings.max_steps}`+(input!==undefined?` · Последний запрос: ${input} вход / ${output??'?'} выход`:'')+(typeof u.cost==='number'?` · $${u.cost.toFixed(6)}`:'');
}
async function branchAt(messageCount, message, send=false) {
  if($('prompt').value.trim()||draftImages.length||draftTexts.length)throw new Error('Сначала отправьте или очистите текущий черновик.');
  const origin=active;
  const branch=await api(`sessions/${origin}/branch`,{message_count:messageCount});
  await openChat(branch.id);
  if(message){$('prompt').value=message.content;draftImages=structuredClone(message.images||[]);renderAttachments();$('prompt').focus();}
  if(send)await control('send');
  else notify('Создана независимая ветка разговора');
}
async function control(kind) {
  $('error').hidden=true;
  const sends=['send','send_now','steer'].includes(kind);
  const snapshot=settings();
  let text=$('prompt').value.trim();
  if(sends){
    if(draftTexts.length)text+='\n\n'+draftTexts.map(f=>`<attached-file name=${JSON.stringify(f.name)}>\n${f.text}\n</attached-file>`).join('\n\n');
    if(!text && draftImages.length)text='Посмотри на приложенные изображения.';
    if(!text)throw new Error('Введите сообщение или приложите файл.');
    if(!snapshot.model)throw new Error('Выберите или введите Model ID.');
    if(kind==='steer'&&draftImages.length)throw new Error('Для изображений используйте Send now или очередь.');
  }
  if(!active){if(!sends)return;active=(await api('sessions',snapshot)).id;}
  await api(`sessions/${active}/actions`,{kind,text:sends?text:'',settings:(sends&&kind!=='steer')||['resume','compact'].includes(kind)?snapshot:undefined,images:sends?draftImages:[]});
  if(sends){$('prompt').value='';draftImages=[];draftTexts=[];renderAttachments();}
  await poll();await listChats();
}
function modeHelp() {
  const mode=$('mode').value;$('writes-label').hidden=mode!=='auto';
  $('mode-help').textContent={chat:'Chat · чтение контекста и ответы',plan:'Plan · исследование и план, без записи',auto:'Auto · инструменты до результата или лимита'}[mode];
}
async function poll(){if(!active||polling)return;polling=true;const id=active;try{const s=await api('sessions/'+id);if(id===active)render(s);}finally{polling=false;}}
function renderAttachments(){
  const all=[...draftImages.map((f,i)=>({name:f.name,remove:()=>draftImages.splice(i,1)})),...draftTexts.map((f,i)=>({name:f.name,remove:()=>draftTexts.splice(i,1)}))];
  $('attachments').replaceChildren(...all.map(f=>{const c=node('span',f.name,'chip'),b=node('button','×');b.onclick=()=>{f.remove();renderAttachments();};c.append(b);return c;}));
}
async function attachFiles(files){
  for(const f of files){
    if(f.type.startsWith('image/')){
      if(!['image/png','image/jpeg','image/webp','image/gif'].includes(f.type))throw new Error('Поддерживаются PNG, JPEG, WebP и GIF.');
      if(draftImages.length>=4||f.size>6_000_000)throw new Error('До 4 изображений, суммарно до 6 MB.');
      const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=reject;r.readAsDataURL(f);});
      if(draftImages.reduce((n,i)=>n+i.data.length,0)+data.length>8_000_000)throw new Error('Суммарный размер изображений больше 6 MB.');
      draftImages.push({name:f.name,mime:f.type,data});
    }else{
      if(f.size>100_000)throw new Error('Текстовый файл должен быть меньше 100 KB. Большие файлы подключайте через папку проекта.');
      const text=await f.text();if(text.includes('\u0000')||text.includes('\ufffd'))throw new Error('Ожидается UTF-8 текст. PDF, офисные документы и архивы пока не поддерживаются.');
      if(draftTexts.reduce((n,t)=>n+t.text.length,0)+text.length>110_000)throw new Error('Слишком много текстовых вложений для одного сообщения.');
      draftTexts.push({name:f.name,text});
    }
    renderAttachments();
  }
}
function renderModelInfo() {
  const model=(catalogs.get($('provider').value)||[]).find(m=>m.id===$('model').value.trim());
  $('model-info').hidden=!model;
  if(!model)return;
  const parts=[model.name||model.id];
  if(model.context_length)parts.push(`Контекст: ${Number(model.context_length).toLocaleString('ru')}`);
  if(model.top_provider?.max_completion_tokens)parts.push(`Ответ до ${model.top_provider.max_completion_tokens}`);
  if(model.architecture?.input_modalities)parts.push('Вход: '+model.architecture.input_modalities.join(', '));
  if(model.supported_parameters)parts.push(model.supported_parameters.includes('tools')?'Инструменты поддерживаются':'Инструменты не указаны в каталоге');
  for(const [key,label] of [['prompt','вход'],['completion','выход']]) {
    const raw=model.pricing?.[key];if(raw!==undefined&&raw!==null&&raw!==''&&Number.isFinite(Number(raw)))parts.push(`$${(Number(raw)*1000000).toLocaleString('en',{maximumFractionDigits:4})} / 1M ${label}`);
  }
  $('model-info').textContent=parts.join(' · ');
}
$('model').oninput=renderModelInfo;
async function showConnections(){
  await loadConfig();
  $('key-fields').replaceChildren(...config.providers.filter(p=>p.id!=='local').map(p=>{
    const div=node('div',undefined,'key-row'),label=node('label',p.name+' · '+p.key_env),input=node('input');
    input.type='password';input.autocomplete='new-password';input.dataset.provider=p.id;input.placeholder=p.configured?'Введите новый ключ для замены':'API key';label.append(input);
    const sources={environment:'Из переменной окружения',system:'Из системного хранилища',memory:'Только в памяти',none:'Ключ не задан'};
    div.append(label,node('small',(sources[p.key_source]||'')+(p.saved?' · сохранён для перезапуска':'')));
    if(p.key_error)div.append(node('p',p.key_error,'muted'));
    const storage=node('label'),persist=node('input');persist.type='checkbox';persist.dataset.persist=p.id;persist.checked=p.saved;persist.disabled=!config.credential_storage;
    storage.append(persist,node('span',config.credential_storage?'Системное хранилище (без отметки — только память, сохранённый ключ удаляется)':'Системное хранилище недоступно; используйте ENV'));div.append(storage);
    const actions=node('div',undefined,'key-actions');
    if(config.credential_storage&&p.configured){const save=node('button','Сохранить текущий ключ');save.onclick=async()=>{save.disabled=true;try{await api(`providers/${p.id}/key`,{key:'',persist:true,use_current:true});await showConnections();notify('Ключ сохранён в системном хранилище');}catch(e){notify(e.message);save.disabled=false;}};actions.append(save);}
    if(p.configured||p.saved){const remove=node('button','Удалить ключ');remove.onclick=async()=>{remove.disabled=true;try{await api(`providers/${p.id}/key`,{key:'',persist:false});await showConnections();notify('Ключ удалён из Studio и системного хранилища. ENV не изменены.');}catch(e){notify(e.message);remove.disabled=false;}};actions.append(remove);}
    div.append(actions);return div;
  }));
  if(!$('connection-dialog').open)$('connection-dialog').showModal();
}
function rootRow(root={alias:'',path:'',writable:false}){
  const div=node('div'),alias=node('input'),path=node('input'),remove=node('button','×'),label=node('label','Запись'),write=node('input');
  alias.placeholder='Имя контекста';alias.value=root.alias;alias.dataset.field='alias';path.placeholder='/абсолютный/путь';path.value=root.path;path.dataset.field='path';write.type='checkbox';write.checked=root.writable;write.dataset.field='writable';label.prepend(write);remove.onclick=()=>div.remove();div.append(alias,path,remove,label);$('root-editor').append(div);
}
function editProject(fresh=false){const p=fresh?{id:'',name:'',instructions:'',roots:[]}:config.projects.find(p=>p.id===$('project').value);editingProject=p.id;$('project-title').value=p.name;$('project-instructions').value=p.instructions;$('root-editor').replaceChildren();p.roots.forEach(rootRow);if(!p.roots.length)rootRow();$('project-dialog').showModal();}
$('context-toggle').onclick=()=>document.querySelector('.context-panel').classList.toggle('open');
$('menu-toggle').onclick=()=>document.querySelector('.sidebar').classList.toggle('open');
$('save-project').onclick=async()=>{try{const roots=[...$('root-editor').children].map(d=>({alias:d.querySelector('[data-field=alias]').value.trim(),path:d.querySelector('[data-field=path]').value.trim(),writable:d.querySelector('[data-field=writable]').checked}));const p=await api('projects',{id:editingProject,name:$('project-title').value.trim(),instructions:$('project-instructions').value,roots});await loadConfig();$('project').value=p.id;renderContext();$('project-dialog').close();if(current&&current.settings.project_id!==p.id)resetChat();else await listChats();notify('Контекст проекта сохранён');}catch(e){notify(e.message);}};
$('save-keys').onclick=async()=>{const button=$('save-keys');button.disabled=true;try{for(const input of $('key-fields').querySelectorAll('input[data-provider]'))if(input.value.trim()){const persist=[...$('key-fields').querySelectorAll('input[data-persist]')].find(p=>p.dataset.persist===input.dataset.provider).checked;await api(`providers/${input.dataset.provider}/key`,{key:input.value.trim(),persist});input.value='';}$('key-fields').replaceChildren();$('connection-dialog').close();await loadConfig();notify('Ключи сохранены с выбранными настройками');}catch(e){notify(e.message);}finally{button.disabled=false;}};
$('load-models').onclick=async()=>{const provider=$('provider').value;$('load-models').disabled=true;try{const v=await api(`providers/${provider}/models`);if($('provider').value!==provider)return;catalogs.set(provider,v.catalog||[]);renderModelInfo();$('models').replaceChildren(...v.models.map(m=>{const o=node('option');o.value=m;return o;}));notify(`${v.models.length} моделей. Выберите Model ID.`);}catch(e){fail(e);}finally{$('load-models').disabled=false;}};
$('provider').onchange=()=>{$('model').value='';$('models').replaceChildren();renderModelInfo();};$('project').onchange=()=>{renderContext();resetChat();};$('mode').onchange=modeHelp;$('new-chat').onclick=resetChat;$('search').oninput=()=>listChats().catch(fail);$('history-folder').onchange=()=>listChats().catch(fail);
$('compact').onclick=()=>control('compact').catch(fail);
$('send').onclick=()=>control('send').catch(fail);$('send-now').onclick=()=>control('send_now').catch(fail);$('steer').onclick=()=>control('steer').catch(fail);$('stop').onclick=()=>control('stop').catch(fail);$('resume').onclick=()=>control('resume').catch(fail);
$('prompt').onkeydown=e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)){e.preventDefault();control('send').catch(fail);}};
$('files').onchange=async e=>{try{await attachFiles(e.target.files);}catch(e){fail(e);}e.target.value='';};
$('prompt').addEventListener('paste',e=>{const files=[...e.clipboardData.files];if(files.length){e.preventDefault();attachFiles(files).catch(fail);}});
$('messages').ondragover=e=>e.preventDefault();$('messages').ondrop=e=>{e.preventDefault();attachFiles(e.dataTransfer.files).catch(fail);};
$('connections').onclick=()=>showConnections().catch(fail);$('new-project').onclick=()=>editProject(true);$('edit-project').onclick=()=>editProject();$('manage-context').onclick=()=>editProject();$('add-root').onclick=()=>rootRow();
$('manage-chat').onclick=()=>{if(!current)return;$('chat-title').value=current.title;$('archive-chat').disabled=current.status==='running'||current.folder==='archived';$('trash-chat').disabled=current.status==='running'||current.folder==='trash';$('restore-chat').disabled=!current.folder||current.folder==='active';$('history-dialog').showModal();};
async function updateHistory(kind,text){if(!active)return;await api(`sessions/${active}/actions`,{kind,text});$('history-dialog').close();if(kind==='move')$('history-folder').value=text;await poll();await listChats();}
$('rename-chat').onclick=()=>updateHistory('rename',$('chat-title').value.trim()).catch(e=>notify(e.message));
$('archive-chat').onclick=()=>updateHistory('move','archived').catch(e=>notify(e.message));
$('trash-chat').onclick=()=>updateHistory('move','trash').catch(e=>notify(e.message));
$('restore-chat').onclick=()=>updateHistory('move','active').catch(e=>notify(e.message));
$('import-chat').onclick=()=>$('import-file').click();
$('import-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>9000000)throw new Error('Файл импорта должен быть меньше 9 MB.');const session=JSON.parse(await file.text());const imported=await api('sessions/import',{session,project_id:$('project').value});$('search').value='';await openChat(imported.id);notify('Разговор импортирован отдельной копией');}catch(e){fail(e);}finally{$('import-file').value='';}};
$('export').onclick=()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=`allpaka-${current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
(async()=>{await loadConfig();await listChats();bindSuggestions();modeHelp();setInterval(()=>poll().catch(fail),650);setInterval(()=>listChats().catch(fail),5000);})().catch(fail);
