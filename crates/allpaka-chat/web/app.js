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
let config, active = null, current = null, draftImages = [], draftTexts = [], editingProject = '', lastMessages = '', lastPlan = '', polling = false;
let layoutFix = false;
const PREFS_KEY = 'allpaka.studio.prefs.v1';
let prefs = {};
try { prefs = JSON.parse(localStorage.getItem(PREFS_KEY) || '{}'); } catch { prefs = {}; }
function savePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({
      project: $('project').value,
      provider: $('provider').value,
      model: $('model').value.trim(),
      mode: $('mode').value,
      steps: Number($('steps').value),
      outputTokens: Number($('output-tokens').value),
      autoCompact: $('auto-compact').checked,
      compactThreshold: Number($('compact-threshold').value),
      writes: $('writes').checked,
      jsonMode: $('json-mode').checked,
      swarm: collectSwarm(),
      layoutFix
    }));
  } catch {}
}
function applyPrefs() {
  if (prefs.project && config.projects.some(p => p.id === prefs.project)) {
    $('project').value = prefs.project;
  }
  const providerOk = config.providers.some(p => p.id === prefs.provider);
  if (providerOk) $('provider').value = prefs.provider;
  if (providerOk && typeof prefs.model === 'string') $('model').value = prefs.model;
  if (['chat', 'plan', 'auto', 'goal', 'swarm'].includes(prefs.mode)) $('mode').value = prefs.mode;
  const steps = Number(prefs.steps);
  if (Number.isFinite(steps) && steps >= 1 && steps <= 50) $('steps').value = steps;
  const out = Number(prefs.outputTokens);
  if (Number.isFinite(out) && out >= 256 && out <= 393216) $('output-tokens').value = out;
  $('auto-compact').checked = prefs.autoCompact !== false;
  const threshold = Number(prefs.compactThreshold);
  if (Number.isFinite(threshold) && threshold >= 4096 && threshold <= 1000000) $('compact-threshold').value = threshold;
  $('writes').checked = prefs.writes === true;
  $('json-mode').checked = prefs.jsonMode === true;
  layoutFix = prefs.layoutFix === true;
  $('layout-toggle')?.classList.toggle('active', layoutFix);
  applySwarm(prefs.swarm);
}
const catalogs = new Map();
const welcome = $('messages').innerHTML;
const stateNames = {idle:'Готов',running:'В работе',paused:'Приостановлен',error:'Требует внимания'};
const commands = [
  {cmd:'/chat',desc:'Чтение контекста и ответы'},
  {cmd:'/plan',desc:'Исследование и план, без записи'},
  {cmd:'/auto',desc:'Инструменты до результата или лимита'},
  {cmd:'/goal',desc:'Автономная работа к цели'},
  {cmd:'/swarm',desc:'Волна независимых агентов и сводный MASTER'},
  {cmd:'/new',desc:'Новый разговор'},
  {cmd:'/compact',desc:'Сжать контекст'},
  {cmd:'/export',desc:'Скачать разговор'},
  {cmd:'/help',desc:'Список команд'},
];
let commandIndex=0,commandMatches=[];
async function api(path, body) {
  const r = await fetch('/api/' + path, {method:body === undefined?'GET':'POST',headers:body === undefined?{}:{'Content-Type':'application/json','X-Allpaka-Client':'studio'},body:body === undefined?undefined:JSON.stringify(body)});
  const value = await r.json(); if (!r.ok) throw new Error(value.error || `HTTP ${r.status}`); return value;
}
async function del(path) {
  const r = await fetch('/api/' + path, {method:'DELETE',headers:{'X-Allpaka-Client':'studio'}});
  const value = await r.json(); if (!r.ok) throw new Error(value.error || `HTTP ${r.status}`); return value;
}
function notify(text) {$('toast').textContent=text;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,4000);}
let undoAction=null,undoTimer=null;
function showUndo(text,action){
  undoAction=action;
  const u=$('undo');
  u.replaceChildren(node('span',text));
  const b=node('button','Вернуть');
  b.onclick=()=>{u.hidden=true;const a=undoAction;undoAction=null;if(a)a().catch(e=>notify(e.message));};
  u.append(b);u.hidden=false;
  clearTimeout(undoTimer);undoTimer=setTimeout(()=>{u.hidden=true;undoAction=null;},7000);
}
function highlightCommand(){
  [...$('command-menu').children].forEach((b,i)=>b.className=i===commandIndex?'active':'');
}
function renderCommandMenu(){
  const text=$('prompt').value;
  if(!text.startsWith('/')||text.includes(' ')){$('command-menu').hidden=true;commandMatches=[];return;}
  const prefix=text.slice(1).toLowerCase();
  commandMatches=commands.filter(c=>c.cmd.slice(1).startsWith(prefix));
  if(!commandMatches.length){$('command-menu').hidden=true;return;}
  commandIndex=Math.min(commandIndex,commandMatches.length-1);
  $('command-menu').replaceChildren(...commandMatches.map((c,i)=>{
    const b=node('button');b.className=i===commandIndex?'active':'';
    b.append(node('span',c.cmd,'cmd'),node('span',c.desc,'desc'));
    b.onmousedown=e=>{e.preventDefault();selectCommand(c.cmd);};
    b.onmouseenter=()=>{commandIndex=i;highlightCommand();};
    return b;
  }));
  $('command-menu').hidden=false;
}
function selectCommand(cmd){
  const p=$('prompt');p.value=cmd+' ';
  $('command-menu').hidden=true;commandMatches=[];
  p.focus();p.setSelectionRange(p.value.length,p.value.length);
}
function fail(e) {$('error').textContent=e.message || String(e);$('error').hidden=false;}
function node(tag,text,className) {const n=document.createElement(tag);if(text !== undefined)n.textContent=text;if(className)n.className=className;return n;}
function toolArg(call){try{const a=JSON.parse(call.function?.arguments||'{}');for(const key of ['path','query','old_text','name']){if(typeof a[key]==='string'&&a[key])return a[key];}}catch{}return '';}
function toolLabel(call){const name=call.function?.name||'Вызов инструмента';const arg=toolArg(call);return arg?`${name} · ${String(arg).slice(0,60)}`:`${name}`;}
function diffLineStats(diff){let add=0,del=0;for(const line of String(diff||'').split('\n')){if(line.startsWith('---')||line.startsWith('+++'))continue;if(line.startsWith('+'))add++;else if(line.startsWith('-'))del++;}return {add,del};}
function providerList(){return (config&&config.providers)||[];}
function providerName(id){const p=providerList().find(p=>p.id===id);return (p&&p.name)||id||'';}
function modelLabel(provider,model){
  const name=providerName(provider);
  const id=String(model||'').trim();
  if(!id)return name?`${name} · Model ID не задан`:'модель не выбрана';
  return name&&name!==id?`${name} · ${id}`:id;
}
function selectedModelLabel(){return modelLabel($('provider').value,$('model').value.trim());}
function contextHeadline(stats){
  const label=selectedModelLabel();
  if(!stats)return `Контекст: новый разговор · модель: ${label}`;
  const pct=stats.context_window?` · ${((stats.estimated_history_tokens/stats.context_window)*100).toFixed(1)}% окна`:'';
  return `Контекст истории ≈ ${Number(stats.estimated_history_tokens).toLocaleString('ru-RU')} токенов${pct} · модель: ${label}`;
}
function lastAssistantModel(s){
  for(let i=s.messages.length-1;i>=0;i--){
    const m=s.messages[i];
    if(m.role==='assistant'&&(m.model||m.provider))return {provider:m.provider||'',model:m.model||''};
  }
  return null;
}
function swarmModelLine(swarm){
  const members=(swarm&&swarm.members)||[];
  if(!members.length)return 'Swarm: участники не заданы — запрос не отправится, пока их не станет хотя бы два.';
  const list=members.map(m=>`${m.label||'без имени'}: ${modelLabel(m.provider,m.model)}`).join('; ');
  const synthesis=(swarm.synthesis_provider||swarm.synthesis_model)?modelLabel(swarm.synthesis_provider||$('provider').value,swarm.synthesis_model):selectedModelLabel();
  return `Swarm: модели участников — ${list}; синтез — ${synthesis}${swarm.critic?'; включён критик-проход':''}. Модель сессии в этом режиме не отвечает.`;
}
function defaultSwarm(){return {members:[],rounds:1,critic:false,max_steps_per_member:2,report_bytes:6000,synthesis_provider:'',synthesis_model:''};}
function clampNum(value,min,max,fallback){const n=Number(value);return Number.isFinite(n)&&n>=min&&n<=max?n:fallback;}
function swarmPanel(){return $('swarm-members');}
function collectSwarm(){
  if(!swarmPanel())return defaultSwarm();
  const members=[...swarmPanel().children].map(row=>({
    label:row.querySelector('[data-swarm=label]').value.trim(),
    role:row.querySelector('[data-swarm=role]').value.trim(),
    provider:row.querySelector('[data-swarm=provider]').value,
    model:row.querySelector('[data-swarm=model]').value.trim(),
    worker:row.querySelector('[data-swarm=worker]').value.trim(),
    worker_project:row.querySelector('[data-swarm=worker_project]').value.trim(),
  }));
  return {
    members,
    rounds:clampNum($('swarm-rounds').value,1,2,1),
    critic:$('swarm-critic').checked,
    max_steps_per_member:clampNum($('swarm-steps').value,1,8,2),
    report_bytes:Math.round(clampNum($('swarm-report-kb').value,1,24,6)*1000),
    synthesis_provider:$('swarm-synth-provider').value,
    synthesis_model:$('swarm-synth-model').value.trim(),
  };
}
function swarmRow(member={}){
  const m={label:'',role:'',provider:'',model:'',...(member||{})};
  const row=node('div',undefined,'swarm-row');
  const label=node('input');label.placeholder='Имя участника';label.value=m.label;label.maxLength=60;label.dataset.swarm='label';label.setAttribute('aria-label','Имя участника');
  const provider=node('input');provider.dataset.swarm='provider';provider.placeholder='ID провайдера на выбранной машине';provider.setAttribute('aria-label','Провайдер участника');provider.maxLength=200;
  const choices=node('datalist');choices.id='swarm-provider-'+crypto.randomUUID();
  choices.replaceChildren(...providerList().map(p=>{const o=node('option',p.name||p.id);o.value=p.id;return o;}));
  provider.setAttribute('list',choices.id);provider.value=m.provider||'';
  const model=node('input');model.placeholder='Model ID';model.value=m.model;model.maxLength=200;model.dataset.swarm='model';model.setAttribute('aria-label','Модель участника');
  const role=node('input');role.placeholder='Роль: что именно проверяет этот агент';role.value=m.role;role.maxLength=4000;role.dataset.swarm='role';role.setAttribute('aria-label','Роль участника');
  const worker=node('input');worker.placeholder='Машина агента: URL Studio (пусто — здесь)';worker.value=m.worker||'';worker.dataset.swarm='worker';worker.setAttribute('aria-label','Машина агента');
  const workerProject=node('input');workerProject.placeholder='ID проекта на машине агента';workerProject.value=m.worker_project||'';workerProject.dataset.swarm='worker_project';workerProject.setAttribute('aria-label','Проект удалённого агента');
  const remove=node('button','×','swarm-remove');remove.type='button';remove.title='Убрать участника';
  remove.onclick=()=>{row.remove();updateSwarmCost();savePrefs();};
  for(const field of [label,provider,model,role,worker,workerProject])field.oninput=field.onchange=()=>{updateSwarmCost();savePrefs();};
  row.append(label,provider,model,role,worker,workerProject,remove,choices);
  return row;
}
function fillSynthProviders(){
  const current=$('swarm-synth-provider').value;
  $('swarm-synth-provider').replaceChildren(...[{id:'',name:'как у сессии'}].concat(providerList()).map(p=>{const o=node('option',p.name||p.id);o.value=p.id;return o;}));
  $('swarm-synth-provider').value=[...$('swarm-synth-provider').options].some(o=>o.value===current)?current:'';
}
function fillSwarmDefaults(announce){
  if(!swarmPanel())return;
  const taken=[...swarmPanel().children].map(row=>row.querySelector('[data-swarm=label]').value.trim()).filter(Boolean);
  const provider=$('provider').value||(providerList()[0]&&providerList()[0].id)||'';
  const model=$('model').value.trim();
  const presets=[
    {label:'scout',role:'инвентаризация: что уже есть в проекте, где лежит и как связано'},
    {label:'risks',role:'риски, пробелы и регрессии: что сломается первым и что не проверено'},
  ];
  const rows=[];
  for(const preset of presets){
    let label=preset.label,suffix=2;
    while(taken.includes(label))label=`${preset.label}${suffix++}`;
    taken.push(label);
    rows.push(swarmRow({...preset,label,provider,model}));
  }
  swarmPanel().append(...rows);
  fillSynthProviders();
  updateSwarmCost();
  savePrefs();
  if(announce)notify('Добавлены два участника: проверьте провайдеров и модели');
}
function applySwarm(saved){
  if(!swarmPanel())return;
  const swarm={...defaultSwarm(),...(saved&&typeof saved==='object'?saved:{})};
  const members=Array.isArray(swarm.members)?swarm.members:[];
  swarmPanel().replaceChildren(...members.map(swarmRow));
  $('swarm-rounds').value=String(clampNum(swarm.rounds,1,2,1));
  $('swarm-steps').value=String(clampNum(swarm.max_steps_per_member,1,8,2));
  $('swarm-report-kb').value=String(clampNum(Math.round((Number(swarm.report_bytes)||6000)/1000),1,24,6));
  $('swarm-critic').checked=swarm.critic===true;
  fillSynthProviders();
  $('swarm-synth-provider').value=providerList().some(p=>p.id===swarm.synthesis_provider)?swarm.synthesis_provider:'';
  $('swarm-synth-model').value=typeof swarm.synthesis_model==='string'?swarm.synthesis_model:'';
  if(!members.length)fillSwarmDefaults(false);
  updateSwarmCost();
}
function updateSwarmCost(){
  if(!swarmPanel())return;
  const swarm=collectSwarm();
  const count=swarm.members.length;
  const rounds=Math.max(1,swarm.rounds);
  const requests=count*rounds+1+(swarm.critic?1:0);
  const problems=[];
  if(count<2)problems.push('нужно не меньше двух участников');
  if(count>6)problems.push('не больше шести участников');
  if(swarm.members.some(m=>!m.label))problems.push('у каждого участника должно быть имя');
  if(swarm.members.some(m=>!m.provider))problems.push('выберите провайдера каждому участнику');
  if(swarm.members.some(m=>!m.model))problems.push('укажите Model ID каждому участнику');
  const split=`участники ${count}×${rounds}${swarm.critic?' + синтез + критик':' + синтез'}`;
  $('swarm-cost').textContent=`Запросов к провайдерам за один ход: ${requests} (${split}). Участники только читают контекст; запись файлов в Swarm недоступна.`+(problems.length?` Не готово: ${problems.join('; ')}.`:'');
  refreshContextModel();
}
function swarmState(report){
  const status=report.status;
  if(status==='done')return {label:'готов',className:'swarm-ok'};
  if(status==='error')return {label:'ошибка',className:'swarm-err'};
  if(status==='cancelled')return {label:'отменён',className:'swarm-err'};
  // The status word is only the phase now; the step it used to carry became a
  // field, so the suffix is assembled here.
  if(status==='running')return {label:report.step>1?`работает · шаг ${report.step}`:'работает',className:'swarm-run'};
  return {label:'в очереди',className:'muted'};
}
function settings() {return {guardrails:$('chat-guardrails-enabled').checked?{input_policy_sha256:$('chat-guardrails-input').value.trim(),output_policy_sha256:$('chat-guardrails-output').value.trim(),action:$('chat-guardrails-action').value}:null,verbosity:$('verbosity').value,project_id:$('project').value,provider:$('provider').value,model:$('model').value.trim(),mode:$('mode').value,max_steps:Number($('steps').value),max_output_tokens:Number($('output-tokens').value),auto_compact:$('auto-compact').checked,compact_threshold:$('compact-threshold').value?Number($('compact-threshold').value):24000,allow_writes:$('writes').checked,json_mode:$('json-mode').checked,swarm:collectSwarm()};}
async function loadConfig() {
  config=await api('config');
  $('project').replaceChildren(...config.projects.map(p=>{const o=node('option',p.name);o.value=p.id;return o;}));
  $('provider').replaceChildren(...config.providers.map(p=>{const o=node('option',p.name+(p.configured?'':' · ключ не задан'));o.value=p.id;return o;}));
  applyPrefs();
  renderContext();
}
function renderContext() {
  const p=config.projects.find(p=>p.id===$('project').value);if(!p)return;
  $('project-name').textContent=p.name;
  $('roots').replaceChildren(...p.roots.map(r=>{const div=node('div',undefined,'root-card');div.append(node('b',(r.repository?'⑂ ':'▱ ')+r.alias),node('small',r.path),node('small',r.writable?'Auto: запись разрешена для папки':'Только чтение'));return div;}));
}
let chatListRequest=0;
let historyOffset=0,historyHasMore=false;
let historyRenderedSignature=null;
const HISTORY_PAGE=30;
let historyTarget=null;
const folderLabels={active:'Чаты',archived:'Архив',trash:'Корзина'};
const folderCounts={active:0,archived:0,trash:0};
let folderCountsRequest=0;
function highlightMatch(text,query){
  const sm=node('small');
  if(!text)return sm;
  const q=(query||'').trim();
  if(!q){sm.textContent=text;return sm;}
  const lower=text.toLowerCase(),ql=q.toLowerCase();
  let i=0;
  while(i<text.length){
    const idx=lower.indexOf(ql,i);
    if(idx<0){sm.append(text.slice(i));break;}
    if(idx>i)sm.append(text.slice(i,idx));
    sm.append(node('mark',text.slice(idx,idx+ql.length)));
    i=idx+ql.length;
  }
  return sm;
}
async function listChats(mode) {
  const version=++chatListRequest;
  const append=mode==='more';
  const statusEl=$('sessions-status');
  const limit=mode==='auto'?Math.max(HISTORY_PAGE,historyOffset||HISTORY_PAGE):HISTORY_PAGE;
  const offset=append?historyOffset:0;
  if(!append&&!$('sessions').children.length){statusEl.textContent='Загрузка…';statusEl.hidden=false;}
  let rows;
  let requestKey;
  try{
    const params=new URLSearchParams({q:$('search').value,project:$('project').value,folder:$('history-folder').value,sort:$('history-sort').value,offset:String(offset),limit:String(limit),bookmarked:String($('history-bookmarked').checked)});
    requestKey=params.toString();
    rows=await api('sessions?'+params);
  }catch(e){
    if(version!==chatListRequest)return;
    statusEl.replaceChildren(node('span','Не удалось загрузить историю: '+e.message));
    const retry=node('button','Повторить');retry.onclick=()=>listChats().catch(fail);
    statusEl.append(retry);statusEl.hidden=false;
    historyRenderedSignature=null;
    $('sessions').replaceChildren();renderHistoryMore(false);
    return;
  }
  if(version!==chatListRequest)return;
  statusEl.hidden=true;
  const signature=JSON.stringify([requestKey,active,rows]);
  if(mode==='auto'&&signature===historyRenderedSignature)return;
  historyRenderedSignature=append?null:signature;
  if(!append){$('sessions').replaceChildren();historyOffset=0;}
  historyOffset=offset+rows.length;
  historyHasMore=rows.length>=limit;
  if(!append&&!rows.length){
    const q=$('search').value.trim();
    const empty=node('div',q?'Ничего не найдено по запросу':$('history-folder').value==='trash'?'Корзина пуста':$('history-folder').value==='archived'?'Архив пуст':'Нет активных чатов','sessions-empty');
    $('sessions').replaceChildren(empty);renderHistoryMore(false);
    return;
  }
  const frag=document.createDocumentFragment();
  for(const r of rows)frag.append(chatRow(r));
  $('sessions').append(frag);
  renderHistoryMore(historyHasMore);
}
function chatRow(r){
  const row=node('div',undefined,'chat-row'+(r.id===active?' active':''));
  const b=node('button',undefined,'chat-open');
  b.append(node('span',r.title),node('small',`${r.provider} · ${stateNames[r.status]||r.status}`));
  if(r.folder&&r.folder!=='active')b.append(node('span',r.folder==='archived'?'Архив':'Корзина','badge'));
  if(r.match_preview)b.append(highlightMatch(r.match_preview,$('search').value));
  b.onclick=()=>openChat(r.id).catch(fail);
  row.append(b);
  const actions=node('div',undefined,'chat-actions');
  const rename=node('button','✎');rename.title='Переименовать';rename.onclick=()=>openHistoryDialog(r);actions.append(rename);
  if(r.folder!=='archived'){const archive=node('button','▸');archive.title='В архив';archive.disabled=r.status==='running';archive.onclick=()=>updateHistory('move','archived',r.id).catch(fail);actions.append(archive);}
  if(r.folder&&r.folder!=='active'){const restore=node('button','↩');restore.title='Восстановить';restore.onclick=()=>updateHistory('move','active',r.id).catch(fail);actions.append(restore);}
  if(r.folder!=='trash'){const trash=node('button','✕');trash.title='В корзину';trash.disabled=r.status==='running';trash.onclick=()=>updateHistory('move','trash',r.id).catch(fail);actions.append(trash);}
  if(r.folder==='trash'){const del=node('button','🗑');del.title='Удалить навсегда';del.onclick=()=>deleteChat(r.id).catch(fail);actions.append(del);}
  row.append(actions);
  return row;
}
function renderHistoryMore(hasMore){
  const box=$('sessions-more');
  if(!hasMore){box.hidden=true;box.replaceChildren();return;}
  const b=node('button','Показать ещё');b.onclick=()=>listChats('more').catch(fail);
  box.replaceChildren(b);box.hidden=false;
}
function updateFolderOptions(){
  const sel=$('history-folder');
  sel.querySelectorAll('option').forEach(o=>{const n=folderCounts[o.value]??0;o.textContent=`${folderLabels[o.value]||o.value} (${n})`;});
}
async function refreshFolderCounts(){
  const request=++folderCountsRequest;
  const project=$('project').value;
  const results=await Promise.all(['active','archived','trash'].map(async f=>{
    const rows=await api('sessions?'+new URLSearchParams({q:'',project,folder:f}));
    return [f,rows.length];
  }));
  if(request!==folderCountsRequest)return;
  for(const [f,n] of results)folderCounts[f]=n;
  updateFolderOptions();
}
function resetChat() {renderPlanCheckpoints(null);renderBackgroundLauncher(null);$('background-tasks').hidden=true;$('conversation-bookmarks').replaceChildren();$('context-statistics-title').textContent=contextHeadline(null);$('context-statistics-body').replaceChildren();$('usage').textContent='';$('compact-summary').textContent='Контекст ещё не сжат.';$('branch-origin').hidden=true;active=null;current=null;lastMessages='';messageCache=[];lastPlan='';pendingSend=null;pendingRev++;$('plan-add').hidden=true;$('messages').innerHTML=welcome;$('title').textContent='Новый разговор';$('error').hidden=true;$('notice').hidden=true;$('queue').replaceChildren();$('plan').replaceChildren(node('li','План появится во время работы','muted'));$('history-folder').value='active';renderStatus('idle');listChats().catch(fail);refreshFolderCounts().catch(()=>{});bindSuggestions();}
function bindSuggestions() {document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;$('mode').value=b.dataset.mode;modeHelp();$('prompt').focus();});}
function renderStatus(status) {
  $('manage-chat').disabled=!active;const archived=!!(current&&current.folder&&current.folder!=='active');$('send').disabled=archived;$('resume').disabled=archived;
  const running=status==='running';$('compact').disabled=!active||running||archived;$('export').disabled=!active;$('status').textContent=stateNames[status]||status;$('status').className='badge '+status;
  $('send').textContent=running?'В очередь ↑':'Отправить ↑';$('send').title=archived?'Восстановите разговор из архива или корзины':'Отправить сообщение';$('steer').hidden=!running;$('send-now').hidden=!running;$('stop').hidden=!running;$('resume').hidden=!['paused','error'].includes(status);
}
async function openChat(id) {
  active=id;lastMessages='';messageCache=[];pendingSend=null;pendingRev++;const s=await api('sessions/'+id);if(active!==id)return;
  $('verbosity').value=s.settings.verbosity||'normal';$('auto-compact').checked=s.settings.auto_compact??true;$('compact-threshold').value=s.settings.compact_threshold??24000;current=s;$('history-folder').value=s.folder||'active';$('project').value=s.settings.project_id;$('provider').value=s.settings.provider;$('model').value=s.settings.model;$('mode').value=s.settings.mode;$('steps').value=s.settings.max_steps;$('output-tokens').value=s.settings.max_output_tokens??8192;$('writes').checked=s.settings.allow_writes;$('json-mode').checked=!!s.settings.json_mode;$('chat-guardrails-enabled').checked=!!s.settings.guardrails;$('chat-guardrails-input').value=s.settings.guardrails?.input_policy_sha256||'';$('chat-guardrails-output').value=s.settings.guardrails?.output_policy_sha256||'';$('chat-guardrails-action').value=s.settings.guardrails?.action||'block';applySwarm(s.settings.swarm);
  savePrefs();
  renderModelInfo();renderContext();modeHelp();render(s);await listChats();
}
function renderContextStatistics(s){
  const stats=s.context_stats;
  $('context-statistics-title').textContent=contextHeadline(stats||null);
  if(!stats){$('context-statistics-body').replaceChildren();return;}
  const fmt=n=>Number(n).toLocaleString('ru-RU');
  const usage=s.usage||{}, actual=usage.prompt_tokens??usage.input_tokens;
  const last=lastAssistantModel(s), chosen=selectedModelLabel();
  const lastLabel=last?modelLabel(last.provider,last.model):'';
  $('context-statistics-body').replaceChildren(...[
    `Модель запроса: ${chosen} — выбор в панели «Модель»; следующее сообщение уйдёт именно туда.`,
    lastLabel?`Последний ответ: ${lastLabel}.`+(lastLabel===chosen?'':' Панель «Модель» изменена — старые ответы остались за прежней моделью.'):'',
    s.settings.mode==='swarm'?swarmModelLine(s.settings.swarm):'',
    `История: ${fmt(stats.messages)} сообщений; сжато: ${fmt(stats.compacted_messages)}.`,
    `До сжатия ≈ ${fmt(stats.original_history_tokens)} токенов; сейчас ≈ ${fmt(stats.estimated_history_tokens)}.`,
    `Автокомпакт: ${stats.auto_compact?'включён':'выключен'}; порог ${fmt(stats.compact_threshold)}; до порога ≈ ${fmt(stats.remaining_before_compact)}.`,
    `Резерв ответа: ${fmt(stats.output_reserve)} токенов.`,
    stats.context_window?`Окно модели: ${fmt(stats.context_window)}; история + резерв ответа ≈ ${stats.history_and_output_percent.toFixed(1)}%.`:'Размер окна этой модели пока не определён.',
    actual!==undefined?`Последний запрос по данным API: вход ${fmt(actual)}, выход ${fmt(usage.completion_tokens??usage.output_tokens??0)} токенов.`:'Фактический расход API пока не получен.',
    'Оценка истории приблизительная, включает сводку и несжатые сообщения. Системные инструкции, схемы инструментов и неотправленный черновик в неё не входят.'
  ].filter(text=>text).map(text=>node('div',text)));
}
function renderUsage(s){
  const u=s.usage||{},input=u.prompt_tokens??u.input_tokens,output=u.completion_tokens??u.output_tokens;
  const last=lastAssistantModel(s);
  $('usage').textContent=`Модель: ${selectedModelLabel()}`+(last?` · Последний ответ: ${modelLabel(last.provider,last.model)}`:'')+` · Шаг ${s.step} / ${s.settings.max_steps}`+(input!==undefined?` · Последний запрос: ${input} вход / ${output??'?'} выход`:'')+(typeof u.cost==='number'?` · $${u.cost.toFixed(6)}`:'');
}
function refreshContextModel(){
  if(!current){$('context-statistics-title').textContent=contextHeadline(null);return;}
  renderContextStatistics(current);
  renderUsage(current);
}
function isToolWithDiff(m){
  if(m.role!=='tool')return null;
  let result;try{result=JSON.parse(m.content);}catch{return null;}
  if(typeof result?.diff!=='string')return null;
  const {add,del}=diffLineStats(result.diff);
  return {result,add,del};
}
function toolCallFor(s,index){
  const m=s.messages[index];
  const related=s.messages.slice(0,index).reverse().find(prev=>(prev.tool_calls||[]).some(c=>c.id===m.tool_call_id));
  const call=related&&(related.tool_calls||[]).find(c=>c.id===m.tool_call_id);
  return call;
}
function filePathFromCall(call){
  try{const a=JSON.parse(call?.function?.arguments||'{}');return typeof a.path==='string'&&a.path?a.path:null;}catch{return null;}
}
function aggStatsSpan(add,del,cls){
  const s=node('span',undefined,cls);
  s.append(node('span',`+${add}`,'agg-add'),node('span',` −${del}`,'agg-del'));
  return s;
}
// Курсор стрима встаёт в конец последнего абзаца: мерцает ровно там, где
// появляется новый текст, а не на новой строке под блоком. Внутри списка
// садится в последний пункт, под блоком кода — на свою строку.
function withStreamCaret(box){
  const caret=node('span',undefined,'stream-caret');
  let tail=box.lastElementChild;
  if(tail&&tail.classList&&tail.classList.contains('text-block'))tail=tail.lastElementChild;
  if(tail&&(tail.tagName==='UL'||tail.tagName==='OL'))tail=tail.lastElementChild;
  if(tail&&tail.tagName==='SECTION')tail=null;
  (tail||box).append(caret);
}
function renderSingleMessage(s,m,index,openKey,detailState){
  const div=node('article',undefined,'message '+m.role);div.dataset.messageIndex=index;
  const header=node('div',m.role==='user'?'ВЫ':m.role==='tool'?'ИНСТРУМЕНТ':'ALLPAKA','role');
  if(m.role==='assistant' && m.reasoning_content && $('analysis-display').value!=='hidden'){
    const reasoning=node('details',undefined,'reasoning');
    const reasoningKey=openKey();
    reasoning.dataset.openKey=reasoningKey;
    reasoning.dataset.message=String(index);
    reasoning.open=detailState.get(reasoningKey)??($('analysis-display').value==='expanded');
    reasoning.append(node('summary',s.status==='running' && index===s.messages.length-1 && !m.content?'Анализ · поступает…':'Анализ'));
    const body=node('div',undefined,'reasoning-body');body.append(StudioContent.render(m.reasoning_content));
    const copy=node('button','Копировать анализ');copy.onclick=async()=>{try{await navigator.clipboard.writeText(m.reasoning_content);notify('Анализ скопирован');}catch{notify('Не удалось скопировать');}};
    body.append(copy);reasoning.append(body);header.append(reasoning);
  }
  if(m.role==='assistant'&&(m.model||m.provider))header.append(node('span',modelLabel(m.provider,m.model),'model-tag'));
  div.append(header);
  if(m.role==='tool'){
    const call=toolCallFor(s,index);
    const name=call?.function?.name||'Инструмент';
    let ok=true,result;try{result=JSON.parse(m.content);if(result&&result.error)ok=false;}catch{}
    const details=node('details');
    const toolKey=openKey();
    details.dataset.openKey=toolKey;
    details.open=detailState.get(toolKey)??false;
    const summary=node('summary');
    summary.append(node('span',ok?'✓':'✗',ok?'tool-ok':'tool-err'),document.createTextNode(' '+name));
    if(typeof result?.diff==='string'){const {add,del}=diffLineStats(result.diff);if(add||del)summary.append(node('span',` +${add} −${del}`,'diff-count'));}
    details.append(summary);
    if(typeof result?.diff==='string'){const {diff,...info}=result;details.append(node('pre',JSON.stringify(info,null,2)));if(diff)details.append(StudioContent.render('```diff\n'+diff+'\n```'));}else details.append(StudioContent.render(m.content));
    div.append(details);
  }else{
    // Пока идёт генерация последнего ответа, в конце текста мерцает курсор,
    // а до первого токена на его месте подпрыгивают три точки.
    const streaming=s.status==='running'&&m.role==='assistant'&&index===s.messages.length-1;
    const placeholder=(m.tool_calls||[]).length?'Вызовы инструментов':m.truncated?'Лимит достигнут до текстового ответа. Увеличьте лимит ответа и продолжите.':streaming?'':'Ответ не получен.';
    const box=StudioContent.render(m.content||placeholder);
    div.append(box);
    if(streaming){
      if(m.content)withStreamCaret(box);
      else{
        const typing=node('div',undefined,'typing');
        typing.setAttribute('role','status');
        for(let d=0;d<3;d++)typing.append(node('span',undefined,'typing-dot'));
        // Текст — только для скринридера: точки ничего не сообщают вслух.
        typing.append(node('span','Ответ формируется…','sr-only'));
        div.append(typing);
      }
    }
  }
  if((m.swarm||[]).length){
    const reports=node('details',undefined,'swarm-reports');
    const reportsKey=openKey();
    reports.dataset.openKey=reportsKey;
    reports.open=detailState.get(reportsKey)??!['running'].includes(s.status);
    const finished=(m.swarm||[]).filter(r=>r.status==='done').length;
    reports.append(node('summary',`Участники Swarm: ${finished}/${m.swarm.length} отчётов`));
    // Only the last Swarm turn can be re-run: a retry rebuilds that message's
    // synthesis, and rewriting an older one would falsify its history.
    const lastSwarm=s.messages.reduce((acc,row,k)=>(row.swarm||[]).length?k:acc,-1);
    const canRetry=index===lastSwarm&&s.status!=='running';
    for(const report of m.swarm){
      const state=swarmState(report);
      const card=node('div',undefined,'swarm-report');
      const head=node('div',undefined,'swarm-report-head');
      head.append(node('b',report.label),node('small',`${report.provider} · ${report.model}${report.round>1?` · волна ${report.round}`:''}`),node('span',state.label,state.className));
      if(canRetry){const again=node('button','Повторить участника');again.title='Перезапустить этого участника и пересобрать итог по всем отчётам';again.onclick=()=>control('retry_member',report.label).catch(fail);head.append(again);}
      card.append(head);
      if(report.error)card.append(node('div',report.error,'muted'));
      if(report.content)card.append(StudioContent.render(report.content));
      reports.append(card);
    }
    div.append(reports);
  }
  if(m.truncated)div.append(node('div','Неполный ответ · достигнут лимит токенов','muted'));
  for(const call of m.incomplete_tool_calls||[]){const d=node('details');const dk=openKey();d.dataset.openKey=dk;d.open=detailState.get(dk)??false;d.append(node('summary','Незавершённый вызов · не выполнен'),node('pre',JSON.stringify(call,null,2)));div.append(d);}
  for(const i of m.images||[]){const img=document.createElement('img');img.alt=i.name;img.src=`data:${i.mime};base64,${i.data}`;div.append(img);}
  for(const call of m.tool_calls||[]){const d=node('details');const ck=openKey();d.dataset.openKey=ck;d.open=detailState.get(ck)??false;d.append(node('summary',toolLabel(call)),node('pre',call.function?.arguments||''));div.append(d);}
  if(s.status!=='running') {
    const actions=node('div',undefined,'message-actions');
    if(m.role==='user'||m.role==='assistant'){
      const bookmark=node('button','Закладка');bookmark.onclick=async()=>{try{
        const existing=current?.id===s.id?current.bookmarks?.find(b=>b.message_index===index):null;
        bookmarkEditing={sessionId:s.id,messageIndex:index};
        $('bookmark-label').value=existing?.label||Array.from(m.content).slice(0,40).join('');
        $('bookmark-edit-dialog').showModal();$('bookmark-label').focus();
      }catch(error){fail(error);}};actions.append(bookmark);
    }
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
}
// The transcript is append-only apart from compaction and branching, so the
// nodes of every message before the first changed one are kept: a poll during
// generation rebuilds the streaming message instead of re-parsing the whole
// conversation. Entries are indexed by message index; a group stores the same
// unit at each of its members.
let messageCache = [],cacheStatus = '';
// Оптимистичный рендер отправки: воркер добавляет сообщение пользователя в
// историю уже после отправки квитанции, поэтому опрос сразу после POST иногда
// застаёт старое состояние. Черновик держится локально, пока сервер не
// подтвердит сообщение тем же текстом.
let pendingSend = null, pendingRev = 0;
function echoedUser(s,text){
  for(const m of s.messages.slice(-4))if(m.role==='user'&&m.content===text)return true;
  return false;
}
// Анимация входа проигрывается один раз на сообщение: опрос каждые 650 мс
// пересобирает растущий ответ, и без этой отметки он бы мигал при каждом
// обновлении. При смене разговора отметки сбрасываются.
let seenSession = '', seenIndexes = new Set();
function firstSight(s,index){
  if(s.id!==seenSession){seenSession=s.id;seenIndexes=new Set();}
  if(seenIndexes.has(index))return false;
  seenIndexes.add(index);return true;
}
function sameMessage(a,b){return a===b||JSON.stringify(a)===JSON.stringify(b);}
function buildMessageNodes(s,detailState){
  const nodes=[],previous=messageCache;
  messageCache=[];
  // Only the messages before the first changed one are reusable, and only while
  // the turn status is what it was: a node reads its predecessors for the tool
  // name and the action buttons, and `running` is baked into the answer
  // placeholder, so a status change invalidates every node.
  let diverged=0;
  while(diverged<s.messages.length&&diverged<previous.length&&sameMessage(s.messages[diverged],previous[diverged].message))diverged++;
  const reusable=s.status===cacheStatus?diverged:0;
  cacheStatus=s.status;
  const held=(start,end)=>end<=reusable&&previous[start]&&previous[start].end===end?previous[start].node:null;
  const keep=(start,end,built)=>{for(let k=start;k<end;k++)messageCache[k]={end,node:built,message:s.messages[k]};};
  let i=0;
  while(i<s.messages.length){
    const m=s.messages[i];
    const diffInfo=isToolWithDiff(m);
    if(diffInfo){
      const group=[{index:i,message:m,...diffInfo}];
      let j=i+1;
      while(j<s.messages.length){
        const next=isToolWithDiff(s.messages[j]);
        if(!next)break;
        group.push({index:j,message:s.messages[j],...next});
        j++;
      }
      if(group.length>=2){
        const kept=held(i,j);
        if(kept){nodes.push(kept);keep(i,j,kept);i=j;continue;}
        const aggKey=`${i}:agg`;
        const aggDetails=node('details',undefined,'file-aggregate');
        aggDetails.dataset.openKey=aggKey;
        aggDetails.open=detailState.get(aggKey)??false;
        const totalAdd=group.reduce((a,g)=>a+g.add,0);
        const totalDel=group.reduce((a,g)=>a+g.del,0);
        const summary=node('summary');
        const headerRow=node('div',undefined,'agg-header');
        const titleSpan=node('span',`Изменено файлов: ${group.length}`,'agg-title');
        const statsSpan=aggStatsSpan(totalAdd,totalDel,'agg-stats');
        const reviewBtn=node('button','Review','agg-review-btn');
        reviewBtn.onclick=(e)=>{e.stopPropagation();aggDetails.open=!aggDetails.open;};
        headerRow.append(titleSpan,statsSpan,reviewBtn);
        summary.append(headerRow);
        aggDetails.append(summary);
        const fileList=node('div',undefined,'agg-file-list');
        const VISIBLE=5;
        const visible=group.slice(0,Math.min(group.length,Math.max(1,VISIBLE)));
        const hidden=group.slice(visible.length);
        for(const g of visible){
          const call=toolCallFor(s,g.index);
          const path=filePathFromCall(call)||'файл';
          const row=node('div',undefined,'agg-file-row');
          row.append(node('span',path,'agg-file-path'));
          row.append(aggStatsSpan(g.add,g.del,'agg-file-stats'));
          fileList.append(row);
        }
        if(hidden.length){
          const moreKey=`${i}:more`;
          const moreDetails=node('details',undefined,'agg-more');
          moreDetails.dataset.openKey=moreKey;
          moreDetails.open=detailState.get(moreKey)??false;
          moreDetails.append(node('summary',`Показать ещё ${hidden.length}`));
          for(const g of hidden){
            const call=toolCallFor(s,g.index);
            const path=filePathFromCall(call)||'файл';
            const row=node('div',undefined,'agg-file-row');
            row.append(node('span',path,'agg-file-path'));
            row.append(aggStatsSpan(g.add,g.del,'agg-file-stats'));
            moreDetails.append(row);
          }
          fileList.append(moreDetails);
        }
        aggDetails.append(fileList);
        const wrap=node('div',undefined,'file-group');
        const syncOpen=()=>wrap.classList.toggle('open',aggDetails.open);
        aggDetails.addEventListener('toggle',syncOpen);
        syncOpen();
        wrap.append(aggDetails);
        for(const g of group){
          let di=0;
          const ok=()=>`${g.index}:${di++}`;
          const div=renderSingleMessage(s,g.message,g.index,ok,detailState);
          div.classList.add('agg-member');
          wrap.append(div);
        }
        if(firstSight(s,i))wrap.classList.add('message-enter');
        nodes.push(wrap);keep(i,j,wrap);
        i=j;
        continue;
      }
    }
    const kept=held(i,i+1);
    if(kept){nodes.push(kept);keep(i,i+1,kept);i++;continue;}
    let detailIndex=0;
    const openKey=()=>`${i}:${detailIndex++}`;
    const built=renderSingleMessage(s,m,i,openKey,detailState);
    if(firstSight(s,i))built.classList.add('message-enter');
    nodes.push(built);keep(i,i+1,built);
    i++;
  }
  return nodes;
}
let memoryProposalSource=null,memoryProposalRequest=0;
let memoryEditing=null,memoryCatalog=[],memoryCatalogRequest=0,memoryExpiryChanged=false,memoryConsolidationSources=null,memoryEditorEpoch=0,memoryCommitInFlight=false;
function memoryConsolidationCatalogPanel(){
  const panel=node('details'),load=node('button','Открыть сохранённые объединения'),body=node('div'),status=node('p');
  panel.append(node('summary','Сохранённые черновики объединения'),load,status,body);
  const scope=$('memory-scope').value,project=$('project').value;
  const current=()=>$('memory-dialog').open&&scope===$('memory-scope').value&&project===$('project').value&&panel.isConnected!==false;
  let offset=0,pageEpoch=0;
  const page=async(start)=>{if(!current()||load.disabled)return;const pageToken=++pageEpoch;load.disabled=true;try{
    const packet=await api('memory/consolidation-proposals?'+new URLSearchParams({project_id:scope,offset:String(start),limit:'20'}));if(!current())return;
    const ids=new Set();if(packet.kind!=='memory_consolidation_catalog'||packet.project_id!==scope||packet.offset!==start||packet.limit!==20||packet.provider_calls!==0||!Number.isInteger(packet.total)||packet.total<0||packet.total>1000||!Array.isArray(packet.proposals)||packet.proposals.length!==Math.min(20,Math.max(0,packet.total-start))||packet.has_more!==(start+packet.proposals.length<packet.total))throw new Error('Некорректный каталог объединений');
    for(const row of packet.proposals){if(!row||typeof row.id!=='string'||!/^[A-Za-z0-9_-]{1,100}$/.test(row.id)||ids.has(row.id)||row.project_id!==scope||!Number.isInteger(row.source_count)||row.source_count<2||row.source_count>20||row.note_count!==1||typeof row.source_sha256!=='string'||!/^[a-f0-9]{64}$/.test(row.source_sha256))throw new Error('Некорректное предложение в каталоге');ids.add(row.id);}
    body.replaceChildren();offset=start;status.textContent='Предложений: '+packet.total;
    for(const row of packet.proposals){const open=node('button',row.id+' · источников: '+row.source_count);body.append(open);open.onclick=async()=>{if(!current()||pageToken!==pageEpoch||open.disabled)return;const epoch=memoryEditorEpoch;open.disabled=true;try{
      const proposal=await api('memory/proposals/'+encodeURIComponent(row.id));if(!current()||pageToken!==pageEpoch||epoch!==memoryEditorEpoch)return;
      const pins=proposal.consolidation_sources,candidate=proposal.notes?.[0],seen=new Set();
      if(proposal.kind!=='memory_consolidation_proposal'||proposal.id!==row.id||proposal.project_id!==scope||proposal.accepted!==false||proposal.source_sha256!==row.source_sha256||!Array.isArray(pins)||pins.length!==row.source_count||pins.some(pin=>{if(!pin||Object.keys(pin).length!==3||typeof pin.id!=='string'||!/^[A-Za-z0-9_-]{1,80}$/.test(pin.id)||seen.has(pin.id)||!Number.isInteger(pin.version)||pin.version<1||pin.version>1000||typeof pin.sha256!=='string'||!/^[a-f0-9]{64}$/.test(pin.sha256))return true;seen.add(pin.id);return false;})||!Array.isArray(proposal.notes)||proposal.notes.length!==1||!candidate||typeof candidate.name!=='string'||!candidate.name.trim()||new TextEncoder().encode(candidate.name).length>200||typeof candidate.content!=='string'||!candidate.content.trim()||new TextEncoder().encode(candidate.content).length>16000)throw new Error('Некорректный черновик объединения');
      showMemory(null);memoryProposalSource={proposal_id:proposal.id,note_index:0};memoryConsolidationSources=pins;$('memory-scope').disabled=true;$('memory-name').value=candidate.name;$('memory-content').value=candidate.content;$('memory-receipt').textContent='Сохранённый черновик: проверьте и отредактируйте текст. При сохранении сервер проверит исходные версии.';
    }catch(error){if(current()&&pageToken===pageEpoch&&epoch===memoryEditorEpoch)status.textContent=error.message;}finally{open.disabled=false;}};}
    if(start>0){const previous=node('button','Предыдущие объединения');previous.onclick=()=>page(Math.max(0,offset-20));body.append(previous);}if(packet.has_more){const next=node('button','Следующие объединения');next.onclick=()=>page(offset+20);body.append(next);}
  }catch(error){if(current())status.textContent=error.message;}finally{load.disabled=false;}};
  load.onclick=()=>page(0);return panel;
}
function memoryConsolidationComposer(){
  const panel=node('details'),select=node('select'),prepare=node('button','Подготовить объединённую заметку'),status=node('p');select.multiple=true;select.size=6;select.setAttribute('aria-label','Исходные заметки для объединения');const scope=$('memory-scope').value,project=$('project').value,notes=memoryCatalog.filter(note=>note.project_id===scope&&!note.removed);for(const note of notes){const option=node('option',note.name+' · версия '+note.version);option.value=note.id;select.append(option);}panel.append(node('summary','Объединить заметки'),node('p','Выберите от 2 до 20 источников. Проверьте и отредактируйте общий текст перед сохранением. Исходники сохранятся.'),select,prepare,status);
  prepare.onclick=()=>{if(!$('memory-dialog').open||scope!==$('memory-scope').value||project!==$('project').value||panel.isConnected===false)return;try{const ids=[...select.selectedOptions].map(option=>option.value);if(ids.length<2||ids.length>20||new Set(ids).size!==ids.length)throw new Error('Выберите от 2 до 20 разных заметок');const chosen=ids.map(id=>notes.find(note=>note.id===id));if(chosen.some(note=>!note||!Number.isInteger(note.version)||note.version<1||typeof note.sha256!=='string'||!/^[a-f0-9]{64}$/.test(note.sha256)||typeof note.content!=='string'))throw new Error('Некорректные источники');const pins=chosen.map(note=>({id:note.id,version:note.version,sha256:note.sha256}));showMemory(null);memoryConsolidationSources=pins;$('memory-scope').disabled=true;$('memory-name').value='Объединённая заметка';const combined=chosen.map(note=>note.name+'\n'+note.content).join('\n\n');$('memory-content').value=new TextEncoder().encode(combined).length<=16000?combined:'';$('memory-receipt').textContent='Объединение '+pins.length+' источников: проверьте текст и сохраните. Сервер проверит исходные версии.';status.textContent='Источников: '+pins.length;return pins;}catch(error){status.textContent=error.message;return null;}};
  const generate=node('button','Подготовить черновик моделью');panel.append(generate);generate.onclick=async()=>{if(generate.disabled)return;const pins=prepare.onclick();if(!pins)return;const epoch=memoryEditorEpoch,selection=JSON.stringify([...select.selectedOptions].map(option=>option.value)),current=()=>epoch===memoryEditorEpoch&&$('memory-dialog').open&&scope===$('memory-scope').value&&project===$('project').value&&panel.isConnected!==false&&selection===JSON.stringify([...select.selectedOptions].map(option=>option.value));generate.disabled=true;status.textContent='Модель готовит предложение…';try{const packet=await api('memory/consolidation-proposals',{settings:{...settings(),project_id:scope,mode:'chat',allow_writes:false},sources:pins});if(!current())return;const candidate=packet.notes?.[0];if(packet.kind!=='memory_consolidation_proposal'||packet.project_id!==scope||typeof packet.id!=='string'||!/^[A-Za-z0-9_-]{1,100}$/.test(packet.id)||packet.accepted!==false||typeof packet.source_sha256!=='string'||!/^[a-f0-9]{64}$/.test(packet.source_sha256)||!Array.isArray(packet.consolidation_sources)||packet.consolidation_sources.length!==pins.length||packet.consolidation_sources.some((source,i)=>!source||['id','version','sha256'].some(key=>source[key]!==pins[i][key]))||!Array.isArray(packet.notes)||packet.notes.length!==1||!candidate||typeof candidate.name!=='string'||!candidate.name.trim()||new TextEncoder().encode(candidate.name).length>200||typeof candidate.content!=='string'||!candidate.content.trim()||new TextEncoder().encode(candidate.content).length>16000)throw new Error('Некорректное предложение объединения');memoryProposalSource={proposal_id:packet.id,note_index:0};memoryConsolidationSources=pins;$('memory-name').value=candidate.name;$('memory-content').value=candidate.content;$('memory-receipt').textContent='Предложение модели: проверьте текст и сохраните. Источники закреплены.';status.textContent='Предложение сохранено: '+packet.id+' · заметка ещё не создана.';}catch(error){if(current())status.textContent=error.message;}finally{generate.disabled=false;}};return panel;
}
function memoryConsolidationPanel(note){
  const panel=node('div');if(!note?.consolidation_sources)return panel;const sources=note.consolidation_sources,seen=new Set();
  if(!Array.isArray(sources)||sources.length<2||sources.length>20||sources.some(source=>{if(!source||Object.keys(source).length!==3||typeof source.id!=='string'||!/^[A-Za-z0-9_-]{1,80}$/.test(source.id)||source.id===note.id||seen.has(source.id)||!Number.isInteger(source.version)||source.version<1||source.version>1000||typeof source.sha256!=='string'||!/^[a-f0-9]{64}$/.test(source.sha256))return true;seen.add(source.id);return false;})){panel.append(node('p','Некорректные ссылки на исходные заметки.'));return panel;}
  const project=$('project').value;const current=()=>memoryEditing===note&&$('memory-dialog').open&&$('project').value===project&&$('memory-scope').value===note.project_id&&panel.isConnected!==false;
  panel.append(node('p','Объединена из '+sources.length+' заметок. Исходные версии сохранены.'));
  for(const source of sources){const row=node('details'),open=node('button','Прочитать исходную версию'),content=node('div');row.append(node('summary',source.id+' · версия '+source.version),node('p',source.sha256),open,content);panel.append(row);open.onclick=async()=>{if(!current()||open.disabled)return;open.disabled=true;try{const snapshot=await api('memory/notes/'+encodeURIComponent(source.id)+'/versions/'+source.version);if(!current())return;if(snapshot.id!==source.id||snapshot.version!==source.version||snapshot.project_id!==note.project_id||snapshot.sha256!==source.sha256||typeof snapshot.content!=='string')throw new Error('Исходная версия не совпадает со ссылкой');content.replaceChildren(node('pre',snapshot.content));}catch(error){if(current())content.replaceChildren(node('p',error.message));}finally{open.disabled=false;}};}
  const check=node('button','Проверить изменения источников'),status=node('div');panel.append(check,status);check.onclick=async()=>{if(!current()||check.disabled)return;check.disabled=true;try{const packet=await api('memory/notes/'+encodeURIComponent(note.id)+'/versions/'+note.version+'/source-status');if(!current())return;if(packet.kind!=='memory_consolidation_source_status'||packet.id!==note.id||packet.version!==note.version||packet.sha256!==note.sha256||packet.project_id!==note.project_id||packet.semantics!=='revision_status'||packet.provider_calls!==0||packet.notes_modified!==false||!Array.isArray(packet.sources)||packet.sources.length!==sources.length)throw new Error('Некорректная проверка источников');for(let i=0;i<sources.length;i++){const row=packet.sources[i],pin=sources[i];if(!row||row.id!==pin.id||row.version!==pin.version||row.sha256!==pin.sha256||!Number.isInteger(row.latest_version)||row.latest_version<pin.version||row.latest_version>1000||typeof row.latest_sha256!=='string'||!/^[a-f0-9]{64}$/.test(row.latest_sha256)||typeof row.removed!=='boolean'||row.status!==(row.removed?'removed':row.latest_version===pin.version?'current':'changed')||row.latest_version===pin.version&&row.latest_sha256!==pin.sha256)throw new Error('Версия источника не совпадает');}status.replaceChildren(node('p','Проверены версии записей. Это не проверка смысла или актуальности фактов.'));for(const row of packet.sources)status.append(node('p',row.id+' · исходная версия '+row.version+' · сейчас '+row.latest_version+' · '+({current:'без изменений',changed:'изменена',removed:'убрана из поиска'}[row.status])));}catch(error){if(current())status.replaceChildren(node('p',error.message));}finally{check.disabled=false;}};
  return panel;
}
function showMemory(note){
  memoryEditorEpoch++;
  memoryProposalSource=null;memoryConsolidationSources=null;memoryEditing=note;$('memory-scope').disabled=!!note;
  $('memory-consolidation-origin').replaceChildren(memoryConsolidationPanel(note));
  const historical=!!note&&(memoryCatalog.find(n=>n.id===note.id)?.version||0)>note.version;
  $('memory-name').readOnly=historical;$('memory-content').readOnly=historical;
  $('memory-expiry').disabled=historical;$('memory-save').disabled=historical;$('memory-remove').disabled=historical;
  $('memory-name').value=note?.name||'';$('memory-content').value=note?.content||'';
  const expiry=note?.expires_ms!=null?new Date(note.expires_ms):null;
  $('memory-expiry').value=expiry&&Number.isFinite(expiry.getTime())?new Date(expiry.getTime()-expiry.getTimezoneOffset()*60000).toISOString().slice(0,16):'';memoryExpiryChanged=false;
  $('memory-version').value=note?.version||1;
  $('memory-load-version').disabled=!note;$('memory-remove').hidden=!note;$('memory-export').hidden=!note;
  $('memory-remove').textContent=note?.removed?'Восстановить заметку':'Убрать из поиска';
  $('memory-receipt').textContent=note?'Сохранена версия '+note.version+(historical?' · историческая версия, только чтение':'')+(note.removed?' · убрана из поиска':note.expires_ms&&note.expires_ms<=Date.now()?' · срок актуальности истёк':''):'Новая заметка';
}
function renderMemoryCatalog(){
  const query=$('memory-search').value.toLowerCase(),rows=[];
  for(const note of memoryCatalog){
    if(note.removed&&!$('memory-show-removed').checked||!(note.name+' '+note.content).toLowerCase().includes(query))continue;
    const open=node('button',note.name+' · версия '+note.version+(note.removed?' · убрана':note.expires_ms&&note.expires_ms<=Date.now()?' · просрочена':''));
    open.onclick=()=>showMemory(note);rows.push(open);
  }
  $('memory-list').replaceChildren(...(rows.length?rows:[node('p','Заметки не найдены.')]));
  $('memory-consolidation-compose').replaceChildren(memoryConsolidationComposer());
}
function memoryExpiryPanel(){
  const panel=node('details'),days=node('input'),load=node('button','Проверить сроки'),body=node('div');days.type='number';days.min=0;days.max=365;days.value='30';days.setAttribute('aria-label','Горизонт проверки памяти в днях');panel.append(node('summary','Сроки памяти'),node('p','Проверка заданных сроков выбранной области. Без автоматического изменения заметок.'),days,load,body);let epoch=0;
  days.oninput=()=>{epoch++;body.replaceChildren();};panel.ontoggle=()=>{if(!panel.open)epoch++;};
  load.onclick=async()=>{if(load.disabled)return;const scope=$('memory-scope').value,horizon=Number(days.value),token=++epoch;const current=()=>token===epoch&&scope===$('memory-scope').value&&$('memory-dialog').open&&panel.isConnected!==false;load.disabled=true;
    try{if(days.value===''||!Number.isInteger(horizon)||horizon<0||horizon>365)throw new Error('Укажите от 0 до 365 дней');const packet=await api('memory/expiry?'+new URLSearchParams({project_id:scope,include_global:'false',horizon_days:String(horizon)}));if(!current())return;
      const counts=packet.counts;if(packet.kind!=='memory_expiry'||packet.project_id!==scope||packet.include_global!==false||packet.horizon_days!==horizon||packet.semantics!=='declared_expiry'||packet.provider_calls!==0||!Number.isSafeInteger(packet.as_of_ms)||packet.as_of_ms<0||packet.horizon_ms!==packet.as_of_ms+horizon*86400000||!counts||['expired','expiring','active','no_expiry','removed'].some(key=>!Number.isInteger(counts[key])||counts[key]<0||counts[key]>2000)||!Array.isArray(packet.notes)||packet.notes.length!==counts.expired+counts.expiring||packet.notes.length>2000)throw new Error('Некорректный обзор сроков');
      const seen=new Set();for(const note of packet.notes){if(note.project_id!==scope||typeof note.id!=='string'||!note.id||seen.has(note.id)||!Number.isSafeInteger(note.version)||note.version<1||typeof note.sha256!=='string'||!/^[a-f0-9]{64}$/.test(note.sha256)||typeof note.name!=='string'||!Number.isSafeInteger(note.expires_ms)||note.expires_ms<0||note.status!==(note.expires_ms<=packet.as_of_ms?'expired':'expiring')||note.expires_ms>packet.horizon_ms)throw new Error('Некорректная заметка в обзоре');seen.add(note.id);}
      body.replaceChildren(node('p','Просрочены: '+counts.expired+' · скоро истекают: '+counts.expiring+' · более поздний срок: '+counts.active+' · без срока: '+counts.no_expiry+' · убраны: '+counts.removed));
      for(const note of packet.notes){const row=node('div'),open=node('button','Прочитать '+note.name),content=node('div');row.append(node('p',note.name+' · '+(note.status==='expired'?'просрочена':'скоро истекает')+' · '+new Date(note.expires_ms).toLocaleString()+' · версия '+note.version),open,content);open.onclick=async()=>{if(!current()||open.disabled)return;open.disabled=true;try{const snapshot=await api('memory/notes/'+encodeURIComponent(note.id)+'/versions/'+note.version);if(!current())return;if(snapshot.id!==note.id||snapshot.project_id!==scope||snapshot.version!==note.version||snapshot.sha256!==note.sha256||snapshot.expires_ms!==note.expires_ms||snapshot.removed||typeof snapshot.content!=='string')throw new Error('Версия заметки не совпадает');content.replaceChildren(node('pre',snapshot.content));}catch(error){if(current())content.replaceChildren(node('p',error.message));}finally{open.disabled=false;}};body.append(row);}
    }catch(error){if(current())body.replaceChildren(node('p',error.message));}finally{load.disabled=false;}
  };return panel;
}
async function loadMemoryCatalog(){const request=++memoryCatalogRequest,scope=$('memory-scope').value;memoryCatalog=[];renderMemoryCatalog();const result=await api('memory/notes?project_id='+encodeURIComponent(scope));if(request!==memoryCatalogRequest||scope!==$('memory-scope').value)return;memoryCatalog=result.notes;renderMemoryCatalog();}
$('memory-open').onclick=async()=>{try{
  const project=node('option',$('project').selectedOptions[0]?.textContent||'Текущий проект');project.value=$('project').value;
  const global=node('option','Общая память');global.value='global';$('memory-scope').replaceChildren(project,global);
  $('memory-search').value='';$('memory-show-removed').checked=false;showMemory(null);
  $('memory-consolidation-catalog').replaceChildren(memoryConsolidationCatalogPanel());$('memory-expiry-review').replaceChildren(memoryExpiryPanel());$('memory-dialog').showModal();await loadMemoryCatalog();await loadMemoryProposals();
}catch(e){fail(e);}};
function memoryExtractionSourcePanel(receipt,sessionId){
  const panel=node('div'),check=node('button','Проверить источник разговора'),status=node('p');panel.append(check,status);
  const project=$('project').value,current=()=>active===sessionId&&project===$('project').value&&$('memory-dialog').open&&panel.isConnected!==false;
  check.onclick=async()=>{if(!current()||check.disabled)return;check.disabled=true;try{
    const packet=await api('memory/proposals/'+encodeURIComponent(receipt.id)+'/source-status');if(!current())return;
    const hash=packet.current_source_sha256,count=packet.current_message_count;
    if(packet.kind!=='memory_extraction_source_status'||packet.proposal_id!==receipt.id||packet.session_id!==sessionId||packet.project_id!==receipt.project_id||packet.message_count!==receipt.message_count||packet.source_sha256!==receipt.source_sha256||packet.semantics!=='extraction_prefix'||packet.provider_calls!==0||packet.notes_modified!==false||typeof packet.running!=='boolean'||!['current','changed','unavailable'].includes(packet.status)||hash!==null&&(typeof hash!=='string'||!/^[a-f0-9]{64}$/.test(hash))||count!==null&&(!Number.isSafeInteger(count)||count<0)||packet.status==='current'&&(hash!==receipt.source_sha256||count<receipt.message_count)||packet.status==='changed'&&(count===null||hash===receipt.source_sha256)||packet.status==='unavailable'&&(hash!==null||count!==null||packet.running))throw new Error('Некорректная проверка источника');
    status.textContent=({current:'Исходная часть разговора совпадает с предложением.',changed:'Исходная часть разговора изменилась. Проверьте предложение заново.',unavailable:'Исходный разговор недоступен для проверки.'}[packet.status])+(packet.running?' Разговор сейчас выполняется.':'')+' Проверены исходные сообщения, а не достоверность фактов.';
  }catch(error){if(current())status.textContent=error.message;}finally{check.disabled=false;}};return panel;
}
async function loadMemoryProposals(){
  const sessionId=active,request=++memoryProposalRequest;
  $('memory-proposals-list').replaceChildren();
  if(!sessionId){$('memory-proposals-status').textContent='Выберите разговор.';return;}
  const result=await api('sessions/'+encodeURIComponent(sessionId)+'/memory-proposals');
  if(request!==memoryProposalRequest||active!==sessionId)return;
  $('memory-proposals-status').textContent=result.proposals.length?'Выберите результат извлечения.':'Сохранённых предложений нет.';
  for(const row of result.proposals){
    const button=node('button','Предложений: '+row.note_count+' · сообщений: '+row.message_count);
    button.onclick=async()=>{try{
      const receipt=await api('memory/proposals/'+encodeURIComponent(row.id));
      if(request!==memoryProposalRequest||active!==sessionId)return;
      $('memory-proposals-list').replaceChildren(memoryExtractionSourcePanel(receipt,sessionId));
      $('memory-proposals-status').textContent='Проверьте предложение и сохраните заметку. Источник: '+receipt.source_sha256.slice(0,12);
      receipt.notes.forEach((note,noteIndex)=>{
        const select=node('button',note.name);
        select.onclick=()=>{
          if(active!==sessionId)return;
          if(![...$('memory-scope').options].some(option=>option.value===receipt.project_id)){fail(new Error('Откройте память проекта исходного разговора.'));return;}
          $('memory-scope').value=receipt.project_id;
          showMemory(null);memoryProposalSource={proposal_id:receipt.id,note_index:noteIndex};
          $('memory-scope').disabled=true;$('memory-name').value=note.name;$('memory-content').value=note.content;
          $('memory-receipt').textContent='Предложение модели — проверьте перед сохранением';
          loadMemoryCatalog().catch(fail);
        };
        $('memory-proposals-list').append(select);
      });
    }catch(e){fail(e);}};
    $('memory-proposals-list').append(button);
  }
}
$('memory-proposals-refresh').onclick=()=>loadMemoryProposals().catch(fail);
$('memory-extract').onclick=async()=>{const sessionId=active;if(!sessionId)return;const button=$('memory-extract');button.disabled=true;try{
  const source=await api('sessions/'+encodeURIComponent(sessionId));
  await api('sessions/'+encodeURIComponent(sessionId)+'/memory-proposals',{settings:{...settings(),project_id:source.settings.project_id,mode:'chat',allow_writes:false},message_count:source.messages.length});
  if(active===sessionId)await loadMemoryProposals();
}catch(e){fail(e);}finally{button.disabled=false;}};
$('memory-close').onclick=()=>$('memory-dialog').close();
$('memory-dialog').onclose=()=>{memoryEditorEpoch++;};
for(const id of ['memory-name','memory-content'])$(id).oninput=()=>{memoryEditorEpoch++;};
$('memory-new').onclick=()=>showMemory(null);
$('memory-scope').onchange=()=>{$('memory-consolidation-catalog').replaceChildren(memoryConsolidationCatalogPanel());$('memory-expiry-review').replaceChildren(memoryExpiryPanel());showMemory(null);loadMemoryCatalog().catch(fail);};
$('memory-search').oninput=renderMemoryCatalog;$('memory-show-removed').onchange=renderMemoryCatalog;
$('memory-expiry').oninput=()=>{memoryExpiryChanged=true;memoryEditorEpoch++;};
async function commitMemory(removed,metadataOnly=false){
  if(memoryCommitInFlight)return;memoryCommitInFlight=true;
  const epoch=memoryEditorEpoch,project=$('project').value,scope=$('memory-scope').value,current=()=>epoch===memoryEditorEpoch&&project===$('project').value&&scope===$('memory-scope').value&&$('memory-dialog').open;
  try{
    const expiry=metadataOnly||!memoryExpiryChanged?memoryEditing?.expires_ms??null:$('memory-expiry').value?new Date($('memory-expiry').value).getTime():null;
    if(expiry!==null&&!Number.isFinite(expiry))throw new Error('Проверьте срок актуальности');
    const note=await api('memory/notes',{id:memoryEditing?.id,project_id:scope,name:metadataOnly?memoryEditing.name:$('memory-name').value,content:metadataOnly?memoryEditing.content:$('memory-content').value,base_version:memoryEditing?.version||0,removed,expires_ms:expiry,...(memoryProposalSource?{proposal_source:memoryProposalSource}:{}),...(memoryConsolidationSources?{consolidation_sources:memoryConsolidationSources}:{})});
    if(!current())return;showMemory(note);notify('Сохранена версия заметки '+note.version);await loadMemoryCatalog();
  }catch(error){if(current())throw error;}finally{memoryCommitInFlight=false;}
}
$('memory-save').onclick=()=>commitMemory(memoryEditing?.removed||false).catch(fail);
$('memory-remove').onclick=()=>commitMemory(!memoryEditing?.removed,true).catch(fail);
$('memory-load-version').onclick=async()=>{if(!memoryEditing)return;try{showMemory(await api('memory/notes/'+encodeURIComponent(memoryEditing.id)+'/versions/'+Number($('memory-version').value)));}catch(e){fail(e);}};
$('memory-export').onclick=()=>{if(!memoryEditing)return;const url=URL.createObjectURL(new Blob([JSON.stringify(memoryEditing,null,2)],{type:'application/json'}));const link=node('a');link.href=url;link.download='memory-'+memoryEditing.id+'-v'+memoryEditing.version+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
let evaluationProject=null,evaluationDataset=null,evaluationSelectedRun=null,evaluationRunStatus=null;
let offlineScoresEpoch=0,offlineCompareEpoch=0;
async function loadOfflineScores(offset=0){
  const epoch=++offlineScoresEpoch,project=evaluationProject,panel=$('offline-scores-result'),filtered=$('offline-scores-dataset-only').checked,dataset=filtered?$('evaluation-dataset').value:null;
  const current=()=>epoch===offlineScoresEpoch&&project===evaluationProject&&$('evaluation-dialog').open&&filtered===$('offline-scores-dataset-only').checked&&(!filtered||dataset===$('evaluation-dataset').value);
  panel.replaceChildren(node('p','Проверка сохранённых оценок…'));
  try{
    if(filtered&&!dataset)throw new Error('Сначала выберите сохранённый набор ниже');
    const query=new URLSearchParams({project_id:project,offset,limit:20});if(filtered)query.set('dataset_id',dataset);
    const catalog=await api('evaluation/score?'+query);
    if(!current())return;
    const rows=[node('p','Проверенных оценок: '+catalog.total+' · пропущено повреждённых: '+catalog.invalid_receipts+(catalog.truncated?' · показана часть каталога':'') )];
    for(const score of catalog.scores){
      if(score.project_id!==project||(filtered&&score.dataset_id!==dataset))throw new Error('Оценка принадлежит другому проекту');
      const section=node('details'),heading=node('summary',score.dataset_id+' · версия '+score.dataset_version+' · '+score.id.slice(-10));section.append(heading);
      const content=node('div');section.append(content);let loaded=false;
      section.ontoggle=async()=>{
        if(!section.open||loaded||!current())return;loaded=true;content.replaceChildren(node('p','Проверка результата…'));
        try{
          const receipt=await api('evaluation/score/'+encodeURIComponent(score.id));
          if(!current())return;
          if(receipt.id!==score.id||receipt.project_id!==project||receipt.kind!=='offline_score')throw new Error('Сохранённая оценка не соответствует выбору');
          const detail=[node('p','Примеров: '+receipt.items.length+' · вызовов модели при оценке: '+receipt.provider_calls)];
          for(const [metric,value] of Object.entries(receipt.mean_scores))detail.push(node('p',(metricLabels[metric]||metric)+': '+value));
          for(const item of receipt.items){const example=node('details');example.append(node('summary',item.sample_id),node('pre',item.output));for(const [metric,value] of Object.entries(item.scores))example.append(node('p',(metricLabels[metric]||metric)+': '+value));detail.push(example);}
          const select=node('div');
          for(const [id,label] of [['offline-score-baseline','Выбрать базовой'],['offline-score-candidate','Выбрать новой']]){
            const button=node('button',label);button.onclick=()=>{if(!current())return;$(id).value=receipt.id;offlineCompareEpoch++;$('offline-scores-comparison').replaceChildren();};select.append(button);
          }
          detail.push(select,savedAnswerJudgePanel(receipt,current));
          content.replaceChildren(...detail);
        }catch(error){if(current()){loaded=false;content.replaceChildren(node('p',error.message));}}
      };
      rows.push(section);
    }
    const nav=node('div');
    if(offset>0){const previous=node('button','Предыдущие оценки');previous.onclick=()=>{if(current())return loadOfflineScores(Math.max(0,offset-20));};nav.append(previous);}
    if(catalog.has_more){const next=node('button','Следующие оценки');next.onclick=()=>{if(current())return loadOfflineScores(offset+20);};nav.append(next);}
    rows.push(nav);panel.replaceChildren(...rows);
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}
}
$('offline-scores-refresh').onclick=()=>loadOfflineScores();
function clearOfflineScores(){offlineScoresEpoch++;offlineCompareEpoch++;$('offline-scores-result').replaceChildren();$('offline-scores-comparison').replaceChildren();$('offline-score-baseline').value='';$('offline-score-candidate').value='';}
$('offline-scores-dataset-only').onchange=()=>{clearOfflineScores();return loadOfflineScores();};
$('offline-scores-compare').onclick=async()=>{
  const epoch=++offlineCompareEpoch,project=evaluationProject,baseline=$('offline-score-baseline').value,candidate=$('offline-score-candidate').value,panel=$('offline-scores-comparison');
  const current=()=>epoch===offlineCompareEpoch&&project===evaluationProject&&$('evaluation-dialog').open;
  try{
    if(!baseline||!candidate)throw new Error('Выберите базовую и новую оценки в раскрытых результатах');
    panel.replaceChildren(node('p','Проверка и сравнение оценок…'));
    const receipts=await Promise.all([api('evaluation/score/'+encodeURIComponent(baseline)),api('evaluation/score/'+encodeURIComponent(candidate))]);
    if(!current())return;
    if(receipts.some((receipt,index)=>receipt.project_id!==project||receipt.id!==[baseline,candidate][index]||receipt.kind!=='offline_score'))throw new Error('Оценки не принадлежат выбранному проекту');
    const comparison=await api('evaluation/score/compare',{baseline_id:baseline,candidate_id:candidate});
    if(!current())return;
    if(comparison.project_id!==project||comparison.baseline_id!==baseline||comparison.candidate_id!==candidate||comparison.kind!=='offline_comparison')throw new Error('Сравнение не соответствует выбору');
    const rows=[node('p','Ухудшений: '+comparison.regressions+' · улучшений: '+comparison.improvements),node('p',comparison.regressions?'Есть ухудшения по отдельным примерам':comparison.eligible?'Есть улучшение без ухудшений':'Ухудшений нет; строгое улучшение не подтверждено')];
    for(const pair of comparison.pairs)rows.push(node('p',pair.sample_id+' · '+(metricLabels[pair.metric]||pair.metric)+': '+pair.baseline+' → '+pair.candidate));
    panel.replaceChildren(...rows);
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}
};
function verifyStudioMatrixJob(job,project){
  if(job.kind!=='experiment_matrix_job'||job.schema_version!==1||job.project_id!==project||typeof job.id!=='string'||!/^[A-Za-z0-9-]{1,80}$/.test(job.id)||!['running','interrupted','failed','cancelled','completed'].includes(job.status)||job.automatic_replay!==false||job.automatic_promotion!==false||!Array.isArray(job.variants)||job.variants.length<2||job.variants.length>16)throw new Error('Задание не соответствует проекту');
  const labels=new Set(),ids=new Set();
  for(const row of job.variants){if(typeof row.label!=='string'||!row.label.trim()||new TextEncoder().encode(row.label).length>200||labels.has(row.label)||row.run_id!==null&&(typeof row.run_id!=='string'||!/^[A-Za-z0-9-]{1,80}$/.test(row.run_id)||ids.has(row.run_id)))throw new Error('Некорректные варианты задания');labels.add(row.label);if(row.run_id)ids.add(row.run_id);}
  if(job.status==='completed'&&(typeof job.matrix_id!=='string'||!/^[A-Za-z0-9-]{1,80}$/.test(job.matrix_id)||job.variants.some(row=>!row.run_id)))throw new Error('Нет подтверждения готовой матрицы');
  if(job.retry_of!=null||job.previous_run_ids!=null){if(typeof job.retry_of!=='string'||!/^[A-Za-z0-9-]{1,80}$/.test(job.retry_of)||job.retry_of===job.id||!Array.isArray(job.previous_run_ids)||job.previous_run_ids.length!==job.variants.length||job.previous_run_ids.some(id=>id!==null&&(typeof id!=='string'||!/^[A-Za-z0-9-]{1,80}$/.test(id))))throw new Error('Некорректная история повтора');}
  return job;
}
async function runServerStudioMatrix(candidates,hooks){
  const frozen=JSON.parse(JSON.stringify(candidates));
  if(frozen.length<2||frozen.length>16)throw new Error('Добавьте от 2 до 16 вариантов');
  const project=frozen[0].request.settings.project_id;
  if(frozen.some(row=>row.request.settings.project_id!==project))throw new Error('Варианты относятся к другому проекту');
  if(hooks.cancelled())throw new Error('Запуск матрицы остановлен');
  let owned=null;
  try{
    const started=verifyStudioMatrixJob(await api('evaluation/matrix-jobs',{project_id:project,variants:frozen.map(row=>({label:row.label,sha256:row.sha256,request:row.request,...(row.promptSha256?{prompt_sha256:row.promptSha256}:{})}))}),project);
    owned=started.id;hooks.progress('задание матрицы',owned,[]);
    let stopping=false;
    while(true){
      if(hooks.detached?.())throw new Error('Наблюдение завершено. Задание продолжает работу в Studio.');
      if(hooks.cancelled()&&!stopping){await api('evaluation/matrix-jobs/'+encodeURIComponent(owned)+'/cancel',{});stopping=true;}
      const job=verifyStudioMatrixJob(await api('evaluation/matrix-jobs/'+encodeURIComponent(owned)),project);
      if(job.id!==owned||job.variants.some((row,index)=>row.label!==frozen[index]?.label)||job.variants.length!==frozen.length)throw new Error('Варианты задания изменились');
      hooks.progress('задание матрицы · '+job.status,owned,[]);
      if(job.status==='running'){await hooks.wait();continue;}
      if(job.status!=='completed'){const error=new Error('Матрица: '+job.status+(job.error?' · '+job.error:''));error.matrixJob=job;throw error;}
      const matrix=await api('evaluation/matrices/'+encodeURIComponent(job.matrix_id));
      if(matrix.kind!=='experiment_matrix'||matrix.schema_version!==1||matrix.id!==job.matrix_id||matrix.project_id!==project||matrix.dataset_sha256!==frozen[0].sha256||matrix.variants.length!==frozen.length||matrix.variants.some((row,index)=>row.label!==job.variants[index].label||row.run_id!==job.variants[index].run_id))throw new Error('Сохранённая матрица не соответствует заданию');
      return {matrix,job};
    }
  }catch(error){error.matrixJobId=owned;throw error;}
}
let evaluationMatrixEpoch=0;
function matrixVariantPairsPanel(variant){
  const panel=node('div'),filter=node('select'),body=node('div');let offset=0;
  const pairs=variant.comparison.pairs||[];
  for(const [value,text] of [['all','Все оценки'],['regressions','Ухудшения'],['improvements','Улучшения'],['unchanged','Без изменений']]){const option=node('option',text);option.value=value;filter.append(option);}filter.value=variant.comparison.regressions?'regressions':'all';filter.setAttribute('aria-label','Изменения оценок варианта');
  const render=()=>{
    const selected=pairs.filter(pair=>filter.value==='all'||filter.value==='regressions'&&pair.delta<0||filter.value==='improvements'&&pair.delta>0||filter.value==='unchanged'&&pair.delta===0);
    const rows=[node('p','Оценок: '+selected.length)];
    for(const pair of selected.slice(offset,offset+20))rows.push(node('p',pair.sample_id+' · '+(metricLabels[pair.metric]||pair.metric)+': '+pair.baseline+' → '+pair.candidate));
    if(!selected.length)rows.push(node('p','Нет оценок, соответствующих фильтру'));
    if(offset){const previous=node('button','Назад');previous.onclick=()=>{offset=Math.max(0,offset-20);render();};rows.push(previous);}
    if(offset+20<selected.length){const next=node('button','Далее');next.onclick=()=>{offset+=20;render();};rows.push(next);}
    body.replaceChildren(...rows);
  };filter.onchange=()=>{offset=0;render();};panel.append(filter,body);render();return panel;
}
function savedExperimentMatrixPanel(receipt){
  const panel=node('div');panel.append(node('h3','Сохранённая матрица · '+receipt.id),node('p',receipt.passed?'Попарных ухудшений нет':'Есть попарные ухудшения'),node('p','Набор: '+receipt.dataset_id+' · версия '+receipt.dataset_version+' · '+receipt.dataset_sha256),node('p','Настройки модели не меняются.'));
  for(const variant of receipt.variants){
    const section=node('details');section.append(node('summary',variant.label+' · '+variant.model),node('p','Ухудшений: '+variant.comparison.regressions+' · улучшений: '+variant.comparison.improvements));
    for(const [metric,value] of Object.entries(variant.mean_scores))section.append(node('p',(metricLabels[metric]||metric)+': '+value));
    const open=node('button','Открыть эксперимент');open.onclick=async()=>{if(evaluationProject!==receipt.project_id||open.disabled)return;open.disabled=true;try{
      const run=await api('evaluation/experiments/'+encodeURIComponent(variant.run_id));
      if(evaluationProject!==receipt.project_id)return;
      if(run.id!==variant.run_id||run.project_id!==receipt.project_id||run.dataset_id!==receipt.dataset_id||run.dataset_version!==receipt.dataset_version||run.dataset_sha256!==receipt.dataset_sha256)throw new Error('Эксперимент не соответствует матрице');
      evaluationSelectedRun=run.id;showEvaluationRun(run);$('evaluation-result').scrollIntoView({block:'center'});
    }catch(error){if(evaluationProject===receipt.project_id)fail(error);}finally{open.disabled=false;}};
    section.append(matrixVariantPairsPanel(variant),open);panel.append(section);
  }
  return panel;
}
let matrixCatalogRequest=0;
$('evaluation-matrices-show').onclick=()=>{
  const project=evaluationProject;
  const show=async offset=>{const version=++matrixCatalogRequest;try{
    const result=await api('evaluation/matrices?'+new URLSearchParams({project_id:project,offset:String(offset),limit:'30'}));
    if(project!==evaluationProject||version!==matrixCatalogRequest)return;
    const rows=[node('p','Сохранённые матрицы. Итог проверяется при открытии.')];
    for(const receipt of result.matrices){const row=node('div'),open=node('button','Открыть');open.onclick=()=>{if(project!==evaluationProject)return;$('evaluation-matrix-id').value=receipt.id;$('evaluation-matrix-open').click();};row.append(node('span',receipt.dataset_id+' · версия '+receipt.dataset_version+' · вариантов '+receipt.variant_count+' · '+receipt.id),open);rows.push(row);}
    if(!result.matrices.length)rows.push(node('p','Нет сохранённых матриц'));
    if(offset>0){const previous=node('button','Назад');previous.onclick=()=>show(Math.max(0,offset-30));rows.push(previous);}
    if(result.has_more){const next=node('button','Далее');next.onclick=()=>show(offset+30);rows.push(next);}
    $('evaluation-matrices-list').replaceChildren(...rows);
  }catch(error){if(project===evaluationProject&&version===matrixCatalogRequest)fail(error);}};
  return show(0);
};
$('evaluation-matrix-open').onclick=async()=>{
  const epoch=++evaluationMatrixEpoch,project=evaluationProject,id=$('evaluation-matrix-id').value.trim();
  try{
    if(!/^[A-Za-z0-9-]{1,80}$/.test(id))throw new Error('Укажите идентификатор матрицы');
    const receipt=await api('evaluation/matrices/'+encodeURIComponent(id));
    if(epoch!==evaluationMatrixEpoch||project!==evaluationProject||id!==$('evaluation-matrix-id').value.trim())return;
    if(receipt.id!==id||receipt.project_id!==project||receipt.kind!=='experiment_matrix'||receipt.schema_version!==1)throw new Error('Матрица не соответствует выбранному проекту');
    $('evaluation-matrix-result').replaceChildren(savedExperimentMatrixPanel(receipt));
  }catch(error){if(epoch===evaluationMatrixEpoch&&project===evaluationProject)$('evaluation-matrix-result').replaceChildren(node('p',error.message));}
};

$('evaluation-matrix-import').onchange=async()=>{
  const epoch=++evaluationMatrixEpoch,project=evaluationProject,panel=$('evaluation-matrix-result');
  try{
    const file=$('evaluation-matrix-import').files[0];if(!file)return;
    if(file.size>32*1024*1024)throw new Error('Отчёт превышает 32 MiB');
    panel.replaceChildren(node('p','Проверка сохранённых экспериментов…'));
    const report=JSON.parse(await file.text());
    const validId=id=>typeof id==='string'&&/^[A-Za-z0-9_-]{1,80}$/.test(id);
    if(report.kind!=='client_experiment_matrix'||!Array.isArray(report.variants)||report.variants.length<1||report.variants.length>16)throw new Error('Нужен отчёт матрицы из SDK или CI');
    const ids=new Set(),labels=new Set();
    for(const row of report.variants){
      if(typeof row.label!=='string'||!row.label.trim()||new TextEncoder().encode(row.label).length>200||labels.has(row.label)||!validId(row.result?.run?.id)||ids.has(row.result.run.id))throw new Error('Некорректные или повторяющиеся варианты');
      ids.add(row.result.run.id);labels.add(row.label);
    }
    const current=()=>epoch===evaluationMatrixEpoch&&project===evaluationProject&&$('evaluation-dialog').open;
    const runs=[];
    for(const row of report.variants){
      const run=await api('evaluation/experiments/'+encodeURIComponent(row.result.run.id));
      if(!current())return;
      if(run.id!==row.result.run.id||run.project_id!==project)throw new Error('Эксперимент отсутствует в выбранном проекте');
      runs.push(run);
    }
    const baseline=runs[0],rows=[node('h3','Варианты: '+runs.length)];
    if(report.error)rows.push(node('p','Отчёт содержит незавершённый запуск. Здесь показаны только сохранённые результаты.'));
    for(let index=0;index<runs.length;index++){
      const run=runs[index];
      if(run.dataset_id!==baseline.dataset_id||run.dataset_version!==baseline.dataset_version||run.dataset_sha256!==baseline.dataset_sha256||JSON.stringify([...run.metrics].sort())!==JSON.stringify([...baseline.metrics].sort()))throw new Error('Варианты используют разные наборы или метрики');
      const section=node('details');section.append(node('summary',report.variants[index].label+' · '+run.settings.model+' · '+(evaluationLabels[run.status]||run.status)));
      section.append(node('p','Набор: '+run.dataset_id+' · версия '+run.dataset_version));
      for(const [metric,value] of Object.entries(run.mean_scores||{}))section.append(node('p',(metricLabels[metric]||metric)+': '+value));
      if(index&&baseline.strict_quality&&run.strict_quality){
        const comparison=await api('evaluation/compare',{baseline_id:baseline.id,candidate_id:run.id});
        if(!current())return;
        section.append(node('p','Ухудшений: '+comparison.regressions+' · улучшений: '+comparison.improvements));
        for(const pair of comparison.pairs)section.append(node('p',pair.sample_id+' · '+(metricLabels[pair.metric]||pair.metric)+': '+pair.baseline+' → '+pair.candidate));
      }else if(index)section.append(node('p','Сравнение недоступно: нужны полностью завершённые результаты.'));
      const open=node('button','Открыть эксперимент');open.onclick=()=>{if(project!==evaluationProject)return;evaluationSelectedRun=run.id;showEvaluationRun(run);$('evaluation-result').scrollIntoView({block:'center'});};section.append(open);rows.push(section);
    }
    if(runs.length>=2&&runs.every(run=>run.strict_quality&&run.status==='completed')){
      const save=node('button','Сохранить матрицу в Studio');save.onclick=async()=>{if(!current()||save.disabled)return;save.disabled=true;try{
        const receipt=await api('evaluation/matrices',{project_id:project,baseline_id:baseline.id,variants:report.variants.map(row=>({label:row.label,run_id:row.result.run.id}))});
        if(!current())return;if(receipt.project_id!==project||receipt.kind!=='experiment_matrix')throw new Error('Не удалось подтвердить сохранённую матрицу');
        $('evaluation-matrix-id').value=receipt.id;panel.replaceChildren(savedExperimentMatrixPanel(receipt));
      }catch(error){if(current())fail(error);}finally{save.disabled=false;}};rows.push(save);
    }
    if(current())panel.replaceChildren(...rows);
  }catch(error){if(epoch===evaluationMatrixEpoch&&project===evaluationProject)panel.replaceChildren(node('p',error.message));}
  finally{if(epoch===evaluationMatrixEpoch)$('evaluation-matrix-import').value='';}
};
let studioMatrixDraft=[],studioMatrixJob=null,studioMatrixRunEpoch=0;
function renderStudioMatrixDraft(){
  const rows=[];for(const candidate of studioMatrixDraft){
    const row=node('div'),label=node('input'),model=node('input'),remove=node('button','Убрать вариант');label.value=candidate.label;label.maxLength=200;label.setAttribute('aria-label','Название варианта');model.value=candidate.request.settings.model;model.setAttribute('aria-label','Модель варианта');
    label.oninput=()=>candidate.label=label.value;model.oninput=()=>candidate.request.settings.model=model.value;
    remove.onclick=()=>{if(studioMatrixJob)return;studioMatrixDraft=studioMatrixDraft.filter(item=>item!==candidate);renderStudioMatrixDraft();};
    for(const control of [label,model,remove])control.disabled=!!studioMatrixJob;
    const template=node('textarea');template.value=candidate.request.prompt_template??candidate.promptPreview??'';template.setAttribute('aria-label','Задание варианта');template.readOnly=!!candidate.request.prompt_ref;template.disabled=!!studioMatrixJob;template.oninput=()=>{if(!candidate.request.prompt_ref)candidate.request.prompt_template=template.value;};
    row.append(label,model,node('p','Набор: '+candidate.request.dataset_id+' · версия '+candidate.request.dataset_version+(candidate.request.prompt_ref?' · задание '+candidate.request.prompt_ref.id+' v'+candidate.request.prompt_ref.version:' · задание из поля')),remove,template);rows.push(row);
  }
  $('matrix-draft-rows').replaceChildren(...rows);$('matrix-draft-add').disabled=!!studioMatrixJob||studioMatrixDraft.length>=16;$('matrix-draft-run').disabled=!!studioMatrixJob||studioMatrixDraft.length<2;$('matrix-draft-stop').disabled=!studioMatrixJob;
}
$('matrix-draft-add').onclick=()=>{try{
  if(studioMatrixJob)return;if(!evaluationDataset||studioMatrixDraft.length>=16)throw new Error('Сохраните набор и добавьте не более 16 вариантов');
  if(promptOrigin&&!evaluationPrompt)throw new Error('Сначала сохраните вариант задания');
  const metrics=['character_bigram_f1','exact_match','contains_reference','json_valid','json_equals','whitespace_token_f1'].filter(metric=>$( {character_bigram_f1:'evaluation-bigram-f1',exact_match:'evaluation-exact',contains_reference:'evaluation-contains',json_valid:'evaluation-json',json_equals:'evaluation-json-equals',whitespace_token_f1:'evaluation-token-f1'}[metric]).checked);
  if(!metrics.length)throw new Error('Выберите хотя бы одну метрику');
  const request={dataset_id:evaluationDataset.id,dataset_version:evaluationDataset.version,settings:{...settings(),project_id:evaluationProject,mode:'chat',allow_writes:false},metrics,...(evaluationPrompt?{prompt_ref:{id:evaluationPrompt.id,version:evaluationPrompt.version}}:{prompt_template:$('evaluation-prompt').value})};
  let number=1;while(studioMatrixDraft.some(candidate=>candidate.label==='Вариант '+number))number++;
  studioMatrixDraft.push(JSON.parse(JSON.stringify({label:'Вариант '+number,sha256:evaluationDataset.sha256,promptPreview:evaluationPrompt?.template,promptSha256:evaluationPrompt?.sha256,request})));renderStudioMatrixDraft();
}catch(error){$('matrix-draft-status').replaceChildren(node('p',error.message));}};
$('matrix-draft-stop').onclick=()=>{if(studioMatrixJob){studioMatrixJob.cancelled=true;$('matrix-draft-status').append(node('p','Остановка запрошена. Новые варианты не запускаются.'));}};
$('matrix-draft-run').onclick=async()=>{
  if(studioMatrixJob)return;const job={cancelled:false,project:evaluationProject,epoch:++studioMatrixRunEpoch};studioMatrixJob=job;renderStudioMatrixDraft();
  const current=()=>studioMatrixJob===job&&evaluationProject===job.project;
  try{
    if(studioMatrixDraft.some(candidate=>candidate.request.settings.project_id!==job.project))throw new Error('Варианты относятся к другому проекту');
    const result=await runServerStudioMatrix(studioMatrixDraft,{cancelled:()=>job.cancelled,detached:()=>!current(),wait:()=>new Promise(resolve=>setTimeout(resolve,1000)),progress:(label,id,results)=>{if(current()){$('matrix-job-id').value=id||'';$('matrix-draft-status').replaceChildren(node('p','Выполняется '+label+' · '+id));}}});
    if(current()){$('evaluation-matrix-id').value=result.matrix.id;$('matrix-draft-status').replaceChildren(savedExperimentMatrixPanel(result.matrix));}
  }catch(error){if(current()){
    const rows=[node('p',error.message)];
    if(error.matrixJobId){$('matrix-job-id').value=error.matrixJobId;rows.push(node('p','Сохранённое задание: '+error.matrixJobId));const open=node('button','Открыть сохранённое задание');open.onclick=()=>{if(evaluationProject===job.project)$('matrix-job-open').click();};rows.push(open);}
    $('matrix-draft-status').replaceChildren(...rows);
  }}finally{if(studioMatrixJob===job){studioMatrixJob=null;renderStudioMatrixDraft();}}
};
renderStudioMatrixDraft();
let studioMatrixJobViewEpoch=0,studioMatrixJobCatalogEpoch=0;
function studioMatrixJobPanel(job,project,epoch){
  verifyStudioMatrixJob(job,project);
  const panel=node('div'),state=node('p','Задание '+job.id+' · '+job.status),message=node('p',job.error||''),refresh=node('button','Обновить состояние');
  const current=()=>evaluationProject===project&&studioMatrixJobViewEpoch===epoch;
  panel.append(state,message,node('p','Закрытие страницы не останавливает задание. После перезапуска продолжение требует явного действия.'));
  if(job.retry_of){panel.append(node('p','Повтор задания: '+job.retry_of));const parent=node('button','Открыть исходное задание');parent.onclick=()=>{if(current()){$('matrix-job-id').value=job.retry_of;return $('matrix-job-open').click();}};panel.append(parent);}
  for(const row of job.variants)panel.append(node('p',row.label+' · '+(row.run_id||'ещё не запущен')));
  async function reload(action){
    if(!current())return;refresh.disabled=true;
    try{if(action)await api('evaluation/matrix-jobs/'+encodeURIComponent(job.id)+'/'+action,{});
      const latest=verifyStudioMatrixJob(await api('evaluation/matrix-jobs/'+encodeURIComponent(job.id)),project);
      if(latest.id!==job.id)throw new Error('ID задания изменился');
      if(current())$('matrix-job-result').replaceChildren(studioMatrixJobPanel(latest,project,++studioMatrixJobViewEpoch));
    }catch(error){if(current())message.textContent=error.message;}finally{refresh.disabled=false;}
  }
  refresh.onclick=()=>reload();panel.append(refresh);
  if(job.status==='running'){const stop=node('button','Остановить задание');stop.onclick=async()=>{if(!current()||stop.disabled)return;stop.disabled=true;await reload('cancel');stop.disabled=false;};panel.append(stop);}
  if(['interrupted','failed'].includes(job.status)){
    panel.append(node('p','Продолжение может вызвать модели для ещё не отправленных вариантов. Уже начатые прерванные вызовы повторно не запускаются.'));
    const resume=node('button','Продолжить готовые и неотправленные варианты');resume.onclick=async()=>{if(!current()||resume.disabled)return;resume.disabled=true;await reload('resume');resume.disabled=false;};panel.append(resume);
  }
  if(['interrupted','failed','cancelled'].includes(job.status)){
    panel.append(node('p','Повтор создаёт новое задание и новые вызовы моделей для неудачных вариантов. Готовые проверенные результаты используются повторно; исходная история сохраняется.'));
    const retry=node('button','Создать новое задание и повторить неудачные варианты');retry.onclick=async()=>{
      if(!current()||retry.disabled)return;retry.disabled=true;
      try{const next=verifyStudioMatrixJob(await api('evaluation/matrix-jobs/'+encodeURIComponent(job.id)+'/retry',{}),project);
        if(next.retry_of!==job.id||next.id===job.id||JSON.stringify(next.previous_run_ids)!==JSON.stringify(job.variants.map(row=>row.run_id)))throw new Error('Новое задание не связано с исходным');
        if(current()){$('matrix-job-id').value=next.id;$('matrix-job-result').replaceChildren(studioMatrixJobPanel(next,project,++studioMatrixJobViewEpoch));}
      }catch(error){if(current())message.textContent=error.message;}finally{retry.disabled=false;}
    };panel.append(retry);
  }
  if(job.status==='completed'){
    const open=node('button','Открыть готовую матрицу');open.onclick=async()=>{
      if(!current()||open.disabled)return;open.disabled=true;
      try{const matrix=await api('evaluation/matrices/'+encodeURIComponent(job.matrix_id));
        if(matrix.id!==job.matrix_id||matrix.kind!=='experiment_matrix'||matrix.schema_version!==1||matrix.project_id!==project||!Array.isArray(matrix.variants)||matrix.variants.length!==job.variants.length||matrix.variants.some((row,index)=>row.run_id!==job.variants[index].run_id||row.label!==job.variants[index].label))throw new Error('Матрица не соответствует заданию');
        if(current()){$('evaluation-matrix-id').value=matrix.id;panel.append(savedExperimentMatrixPanel(matrix));}
      }catch(error){if(current())message.textContent=error.message;}finally{open.disabled=false;}
    };panel.append(open);
  }
  return panel;
}
$('matrix-job-open').onclick=async()=>{
  const id=$('matrix-job-id').value.trim(),project=evaluationProject,epoch=++studioMatrixJobViewEpoch;
  try{if(!/^[A-Za-z0-9-]{1,80}$/.test(id))throw new Error('Укажите корректный ID задания');
    const job=verifyStudioMatrixJob(await api('evaluation/matrix-jobs/'+encodeURIComponent(id)),project);
    if(job.id!==id)throw new Error('ID задания изменился');
    if(evaluationProject===project&&studioMatrixJobViewEpoch===epoch&&$('matrix-job-id').value.trim()===id)$('matrix-job-result').replaceChildren(studioMatrixJobPanel(job,project,epoch));
  }catch(error){if(evaluationProject===project&&studioMatrixJobViewEpoch===epoch)$('matrix-job-result').replaceChildren(node('p',error.message));}
};
async function showStudioMatrixJobs(offset=0){
  const project=evaluationProject,epoch=++studioMatrixJobCatalogEpoch;
  try{const result=await api('evaluation/matrix-jobs?project_id='+encodeURIComponent(project)+'&offset='+offset+'&limit=30');
    if(evaluationProject!==project||studioMatrixJobCatalogEpoch!==epoch)return;
    if(!Array.isArray(result.jobs)||result.jobs.length>30||result.offset!==offset||typeof result.has_more!=='boolean')throw new Error('Некорректный список заданий');
    const rows=[];for(const job of result.jobs){verifyStudioMatrixJob(job,project);const open=node('button',job.id+' · '+job.status);open.onclick=()=>{if(evaluationProject!==project||studioMatrixJobCatalogEpoch!==epoch)return;$('matrix-job-id').value=job.id;$('matrix-job-open').click();};rows.push(open);}
    if(!rows.length)rows.push(node('p','Сохранённых заданий нет'));
    if(offset){const previous=node('button','Предыдущие задания');previous.onclick=()=>{if(evaluationProject===project&&studioMatrixJobCatalogEpoch===epoch)return showStudioMatrixJobs(Math.max(0,offset-30));};rows.push(previous);}
    if(result.has_more&&offset+30<=1000){const next=node('button','Следующие задания');next.onclick=()=>{if(evaluationProject===project&&studioMatrixJobCatalogEpoch===epoch)return showStudioMatrixJobs(offset+30);};rows.push(next);}
    $('matrix-jobs-list').replaceChildren(...rows);
  }catch(error){if(evaluationProject===project&&studioMatrixJobCatalogEpoch===epoch)$('matrix-jobs-list').replaceChildren(node('p',error.message));}
}
$('matrix-jobs-show').onclick=()=>showStudioMatrixJobs();
let evaluationPrompt=null,promptOrigin=null,promptPreviewEpoch=0;
function collectPromptMessages(){return [...$('prompt-messages-list').children].map(row=>({role:row.messageRole,content:row.querySelector('textarea').value}));}
function showPromptMessages(messages){
  const rows=messages.map((message,index)=>{const row=node('div'),content=node('textarea');row.messageRole=message.role;content.value=message.content;content.maxLength=16000;const label=node('label',message.role==='user'?'User — вопрос':'Assistant — пример ответа');label.append(content);row.append(label);if(index%2===0&&index<messages.length-1){const remove=node('button','Удалить пример диалога');remove.type='button';remove.onclick=()=>{const current=collectPromptMessages();current.splice(index,2);showPromptMessages(current);};row.append(remove);}return row;});$('prompt-messages-list').replaceChildren(...rows);
}
function updatePromptFormat(){const chat=$('prompt-format').value==='messages';$('prompt-messages-editor').hidden=!chat;$('evaluation-prompt').disabled=chat;if(chat&&!$('prompt-messages-list').children.length)showPromptMessages([{role:'user',content:'{{input}}'}]);}
$('prompt-format').onchange=updatePromptFormat;
$('prompt-messages-add-pair').onclick=()=>{const messages=collectPromptMessages();if(messages.length>=15){notify('Допускается не больше семи примеров диалога.');return;}messages.splice(messages.length-1,0,{role:'user',content:'Пример вопроса'},{role:'assistant',content:'Пример ответа'});showPromptMessages(messages);};
function showPrompt(prompt){
  promptPreviewEpoch++;$('prompt-preview-result').replaceChildren();$('prompt-playground-result').replaceChildren();
  evaluationPrompt=prompt;promptOrigin=prompt?.origin||null;
  for(const id of ['playground-has-reference','playground-exact','playground-contains','playground-token-f1','playground-bigram-f1','playground-json','playground-json-equals'])$(id).checked=false; $('playground-reference').value='';
  $('prompt-format').value=prompt?.messages?'messages':'text';showPromptMessages(prompt?.messages||[{role:'user',content:'{{input}}'}]);updatePromptFormat();
  $('prompt-history').replaceChildren();
  $('prompt-name').value=prompt?.name||'';$('prompt-version').value=prompt?.version||1;
  $('evaluation-prompt').value=prompt?.template||'{{input}}';
  $('prompt-system').value=prompt?.system||'Answer the supplied evaluation sample.';
  $('prompt-receipt').textContent=prompt?'Проверка использует сохранённую версию '+prompt.version+'. Изменения в полях сначала нужно сохранить.'+(prompt.origin?' Источник: '+prompt.origin.id+' · версия '+prompt.origin.version+'.':''):'Проверка использует задание из поля ниже и стандартную системную инструкцию.';
}
async function previewSelectedPrompt(){
  const prompt=evaluationPrompt,project=evaluationProject,epoch=++promptPreviewEpoch,input=$('prompt-preview-input').value,contextsText=$('prompt-preview-contexts').value,panel=$('prompt-preview-result');
  if(!prompt){panel.replaceChildren(node('p','Сначала загрузите сохранённую версию промпта.'));return;}
  const current=()=>epoch===promptPreviewEpoch&&prompt===evaluationPrompt&&project===evaluationProject&&project===$('project').value&&$('evaluation-dialog').open&&input===$('prompt-preview-input').value&&contextsText===$('prompt-preview-contexts').value;
  try{
    const receipt=await api('evaluation/prompts/'+encodeURIComponent(prompt.id)+'/versions/'+prompt.version+'/preview',{project_id:project,input,contexts:contextsText.split(/\n\s*\n/).filter(Boolean)});if(!current())return;
    if(receipt.kind!=='prompt_preview'||receipt.project_id!==project||receipt.prompt_id!==prompt.id||receipt.prompt_version!==prompt.version||receipt.prompt_sha256!==prompt.sha256||receipt.provider_calls!==0||receipt.saved!==false||!Array.isArray(receipt.messages)||receipt.messages.length!==(prompt.messages?.length||1)+1||receipt.messages[0].role!=='system'||receipt.messages.slice(1).some((m,i)=>m.role!==(prompt.messages?.[i].role||'user'))||receipt.messages.some(m=>typeof m.content!=='string'))throw new Error('Предпросмотр не совпадает с сохранённым промптом.');
    panel.replaceChildren(node('p','Версия '+receipt.prompt_version+' · SHA-256: '+receipt.prompt_sha256),...receipt.messages.flatMap(m=>[node('p',m.role==='system'?'System':m.role==='user'?'User':'Assistant'),node('pre',m.content)]));
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}
}
$('prompt-preview-show').onclick=()=>previewSelectedPrompt();
for(const id of ['prompt-preview-input','prompt-preview-contexts'])$(id).oninput=()=>{promptPreviewEpoch++;$('prompt-preview-result').replaceChildren();};
let promptPlaygroundStarting=false;
function playgroundEvaluationInput(){
  const metrics=[];for(const [id,metric]of [['playground-exact','exact_match'],['playground-contains','contains_reference'],['playground-token-f1','whitespace_token_f1'],['playground-bigram-f1','character_bigram_f1'],['playground-json','json_valid'],['playground-json-equals','json_equals']])if($(id).checked)metrics.push(metric);
  return {metrics,expected_output:$('playground-has-reference').checked?$('playground-reference').value:null};
}
async function startSelectedPromptPlayground(){
  const prompt=evaluationPrompt,project=evaluationProject,panel=$('prompt-playground-result');if(promptPlaygroundStarting)return;
  if(!prompt){panel.replaceChildren(node('p','Сначала загрузите сохранённую версию промпта.'));return;}
  if(project!==$('project').value||!$('evaluation-dialog').open||prompt.project_id!==project){panel.replaceChildren(node('p','Выбранный промпт не соответствует текущему проекту.'));return;}
  const evaluation=playgroundEvaluationInput(),evaluationSignature=JSON.stringify(evaluation);
  if(evaluation.metrics.some(m=>m!=='json_valid')&&evaluation.expected_output===null){panel.replaceChildren(node('p','Для выбранных метрик укажите эталонный ответ.'));return;}
  const contextsText=$('prompt-preview-contexts').value;
  const body={settings:{...settings(),project_id:project,mode:'chat',allow_writes:false},prompt_ref:{id:prompt.id,version:prompt.version},prompt_sha256:prompt.sha256,input:$('prompt-preview-input').value,contexts:contextsText.split(/\n\s*\n/).filter(Boolean),...evaluation};
  const current=()=>prompt===evaluationPrompt&&project===evaluationProject&&project===$('project').value&&$('evaluation-dialog').open&&body.input===$('prompt-preview-input').value&&contextsText===$('prompt-preview-contexts').value&&evaluationSignature===JSON.stringify(playgroundEvaluationInput());
  promptPlaygroundStarting=true;$('prompt-playground-start').disabled=true;
  try{
    const run=await api('evaluation/playground',body);
    if(typeof run.id!=='string'||!/^[a-f0-9-]{1,80}$/.test(run.id)||run.project_id!==project||run.playground!==true||run.prompt_snapshot?.id!==prompt.id||run.prompt_snapshot?.version!==prompt.version||run.prompt_snapshot?.sha256!==prompt.sha256||!Array.isArray(run.items)||run.items.length!==1||!Array.isArray(run.metrics)||run.metrics.length!==body.metrics.length||JSON.stringify([...run.metrics].sort())!==JSON.stringify([...body.metrics].sort())||run.strict_quality!==false)throw new Error('Ответ запуска отличается от выбранного промпта.');
    if(current()){evaluationSelectedRun=run.id;showEvaluationRun(run);panel.replaceChildren(node('p','Запуск '+run.id+' сохранён. Результат доступен в истории запусков.'));await loadEvaluationRuns();}
    else notify('Пробный запуск '+run.id+' сохранён в истории проекта.');
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}finally{promptPlaygroundStarting=false;$('prompt-playground-start').disabled=false;}
}
$('prompt-playground-start').onclick=()=>startSelectedPromptPlayground();
async function loadPrompts(selected=''){
  const result=await api('evaluation/prompts?project_id='+encodeURIComponent(evaluationProject));
  const blank=node('option','Без сохранённой версии');blank.value='';
  $('prompt-library').replaceChildren(blank,...result.prompts.map(p=>{const o=node('option',p.name+' · последняя версия '+p.version);o.value=p.id;o.dataset.version=p.version;return o;}));
  $('prompt-library').value=selected;
}
$('prompt-new').onclick=()=>{showPrompt(null);$('prompt-library').value='';};
$('prompt-fork').onclick=()=>{promptPreviewEpoch++;$('prompt-preview-result').replaceChildren();if(!evaluationPrompt){notify('Сначала откройте сохранённую версию');return;}promptOrigin={id:evaluationPrompt.id,version:evaluationPrompt.version,sha256:evaluationPrompt.sha256};evaluationPrompt=null;$('prompt-library').value='';$('prompt-name').value+=' — вариант';$('prompt-version').value=1;$('prompt-receipt').textContent='Сохраните вариант перед запуском.';};
async function openPrompt(){const id=$('prompt-library').value;if(!id){showPrompt(null);return;}showPrompt(await api('evaluation/prompts/'+encodeURIComponent(id)+'/versions/'+Number($('prompt-version').value)));}
async function showPromptHistory(id,offset=0){
  const result=await api('evaluation/prompts/'+encodeURIComponent(id)+'/versions?offset='+offset+'&limit=20');
  const rows=[node('p','Сохранённых версий: '+result.total)];
  for(const version of result.versions){
    const row=node('div'),open=node('button','Открыть версию '+version.version);
    open.onclick=async()=>{try{const prompt=await api('evaluation/prompts/'+encodeURIComponent(id)+'/versions/'+version.version);showPrompt(prompt);$('prompt-library').value=id;}catch(e){fail(e);}};
    row.append(open,node('span',' '+version.name+' · '+version.sha256.slice(0,12)));rows.push(row);
  }
  for(const [label,next] of [['Предыдущие',offset-20],['Следующие',offset+20]]){
    if(next<0||next>=result.total)continue;
    const page=node('button',label);page.onclick=()=>showPromptHistory(id,next).catch(fail);rows.push(page);
  }
  $('prompt-history').replaceChildren(...rows);
}
$('prompt-history-open').onclick=()=>{const id=evaluationPrompt?.id||$('prompt-library').value;if(!id){notify('Сначала откройте сохранённое задание');return;}showPromptHistory(id).catch(fail);};
$('prompt-load').onclick=()=>openPrompt().catch(fail);
$('prompt-library').onchange=()=>{const o=$('prompt-library').selectedOptions[0];$('prompt-version').value=o?.dataset.version||1;openPrompt().catch(fail);};
$('prompt-save').onclick=async()=>{try{const p=await api('evaluation/prompts',{id:evaluationPrompt?.id,project_id:evaluationProject,name:$('prompt-name').value,base_version:evaluationPrompt?.version||0,origin:promptOrigin,template:$('prompt-format').value==='messages'?'':$('evaluation-prompt').value,messages:$('prompt-format').value==='messages'?collectPromptMessages():null,system:$('prompt-system').value});showPrompt(p);await loadPrompts(p.id);notify('Сохранена версия задания '+p.version);}catch(e){fail(e);}};
const evaluationLabels={running:'Выполняется',completed:'Завершён',failed:'Ошибка',cancelled:'Отменён',pending:'Ожидает',interrupted:'Прерван',token_limit:'Лимит ответа',step_limit:'Лимит шагов',no_progress:'Приостановлен после повторяющихся ошибок',goal_incomplete:'Цель не завершена',timeout:'Истекло время ожидания'};
const metricLabels={character_bigram_f1:'Сходство соседних символов (F1)',exact_match:'Точное совпадение',contains_reference:'Содержит ожидаемый текст',json_valid:'Корректный JSON',json_equals:'JSON совпадает с эталоном',whitespace_token_f1:'Совпадение слов (F1)'};
function sampleEditor(sample={}){
  const row=node('div');row.className='evaluation-sample';row.sampleId=sample.id||'sample-'+crypto.randomUUID();row.sampleMetadata=sample.metadata||{};
  const input=node('textarea');input.className='sample-input';input.placeholder='Вопрос или задание';input.value=sample.input||'';
  const expected=node('textarea');expected.className='sample-expected';expected.placeholder='Ожидаемый ответ (необязательно для проверки JSON)';expected.value=sample.expected_output??'';
  const contexts=node('textarea');contexts.className='sample-contexts';contexts.placeholder='Справочные материалы; разделяйте пустой строкой';contexts.value=(sample.contexts||[]).join('\n\n');row.originalContexts=sample.contexts||[];row.originalContextsText=contexts.value;
  const reference=node('input');reference.type='checkbox';reference.className='sample-reference';reference.checked=sample.expected_output!=null;expected.oninput=()=>{reference.checked=true;};const referenceLabel=node('label','Есть ожидаемый ответ');referenceLabel.className='sample-reference-label';referenceLabel.prepend(reference);
  const remove=node('button','Убрать пример');remove.type='button';remove.onclick=()=>row.remove();
  const inputLabel=node('label','Вопрос или задание');inputLabel.append(input);
  const expectedLabel=node('label','Ожидаемый ответ');expectedLabel.append(expected);
  const contextsLabel=node('label','Справочные материалы');contextsLabel.append(contexts);
  row.append(inputLabel,referenceLabel,expectedLabel,contextsLabel,remove);return row;
}
let evaluationDatasetOrigin=null;
function showDatasetOrigin(){
  const origin=evaluationDatasetOrigin;
  $('evaluation-origin-info').textContent=origin?'Источник: '+origin.id+' · версия '+origin.version+' · SHA-256: '+origin.sha256:'';
}
function showDataset(dataset){
  evaluationDatasetOrigin=dataset?.origin||null;
  showDatasetOrigin();
  evaluationDataset=dataset;$('evaluation-name').value=dataset?.name||'';$('evaluation-version').value=dataset?.version||1;
  $('evaluation-samples').replaceChildren(...(dataset?.samples||[{}]).map(sampleEditor));
}
let evaluationDatasetCatalogEpoch=0;
async function loadEvaluationDatasets(selectId){
  const epoch=++evaluationDatasetCatalogEpoch,project=evaluationProject;
  const result=await api('evaluation/datasets?project_id='+encodeURIComponent(project));
  if(epoch!==evaluationDatasetCatalogEpoch||project!==evaluationProject||project!==$('project').value||!$('evaluation-dialog').open)return;
  const options=result.datasets.map(d=>{const option=node('option',d.name+' · версия '+d.version);option.value=d.id;option.dataset.version=d.version;return option;});
  const blank=node('option','Выберите набор');blank.value='';
  const selected=selectId||$('evaluation-dataset').value;
  $('evaluation-dataset').replaceChildren(blank,...options);
  if(selected&&result.datasets.some(item=>item.id===selected))$('evaluation-dataset').value=selected;
}
let datasetLifecycleEpoch=0,datasetLifecycleArchived=false;
async function loadDatasetLifecycle(archived=false,offset=0){
  datasetLifecycleArchived=archived;const query=$('dataset-lifecycle-search').value.trim();
  const epoch=++datasetLifecycleEpoch,project=evaluationProject,panel=$('dataset-lifecycle-result');
  const current=()=>epoch===datasetLifecycleEpoch&&query===$('dataset-lifecycle-search').value.trim()&&project===evaluationProject&&project===$('project').value&&$('evaluation-dialog').open;
  try{
    const result=await api('evaluation/datasets?'+new URLSearchParams({project_id:project,archived:String(archived),offset:String(offset),limit:'20',q:query}));
    if(!current())return;
    if(!Array.isArray(result.datasets)||result.datasets.length>20||!Number.isSafeInteger(result.total)||result.total<result.datasets.length||result.offset!==offset||result.limit!==20||result.has_more!==(offset+result.datasets.length<result.total)||result.datasets.some(item=>item.project_id!==project||item.archived!==archived||!Number.isSafeInteger(item.version)||item.version<1||!Number.isSafeInteger(item.lifecycle_revision)||item.lifecycle_revision<0))throw new Error('Получен несовместимый каталог наборов.');
    const rows=[node('p',archived?'Архивные наборы':'Действующие наборы')];
    for(const item of result.datasets){
      const row=node('details');row.append(node('summary',item.name+' · версия '+item.version+' · примеров: '+item.samples));
      const action=node('button',archived?'Восстановить набор':'Архивировать набор'),status=node('p');let busy=false,committed=false;
      action.onclick=async()=>{
        if(!current()||busy)return;busy=true;action.disabled=true;
        try{
          const receipt=await api('evaluation/datasets/'+encodeURIComponent(item.id)+'/lifecycle',{project_id:project,base_version:item.version,base_revision:item.lifecycle_revision,archived:!archived});
          if(!current())return;
          if(receipt.id!==item.id||receipt.project_id!==project||receipt.version!==item.version||receipt.sha256!==item.sha256||receipt.archived!==!archived||receipt.lifecycle_revision!==item.lifecycle_revision+1||receipt.snapshots_retained!==true)throw new Error('Ответ изменения набора не соответствует выбранной версии.');
          committed=true;await loadEvaluationDatasets();if(current())await loadDatasetLifecycle(archived,offset);
        }catch(error){if(current())status.textContent=error.message;}finally{if(current()&&!committed){busy=false;action.disabled=false;}}
      };
      row.append(action,status);rows.push(row);
    }
    if(!result.datasets.length)rows.push(node('p','Наборов нет.'));
    if(offset>0||result.has_more){const previous=node('button','Предыдущие наборы'),next=node('button','Следующие наборы');previous.disabled=offset===0;next.disabled=!result.has_more;
      previous.onclick=()=>{if(current()&&!previous.disabled)return loadDatasetLifecycle(archived,Math.max(0,offset-20));};next.onclick=()=>{if(current()&&!next.disabled)return loadDatasetLifecycle(archived,offset+20);};rows.push(previous,next);}
    panel.replaceChildren(...rows);
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}
}
$('dataset-lifecycle-search').oninput=()=>{datasetLifecycleEpoch++;$('dataset-lifecycle-result').replaceChildren();};
$('dataset-lifecycle-find').onclick=()=>loadDatasetLifecycle(datasetLifecycleArchived);
$('dataset-active-show').onclick=()=>loadDatasetLifecycle(false);
$('dataset-archive-show').onclick=()=>loadDatasetLifecycle(true);
let datasetVersionsEpoch=0;
async function loadDatasetVersions(offset=0){
  const origin=evaluationDataset,epoch=++datasetVersionsEpoch,project=evaluationProject,id=$('evaluation-dataset').value||evaluationDataset?.id,panel=$('dataset-versions-result');
  const current=()=>origin===evaluationDataset&&epoch===datasetVersionsEpoch&&project===evaluationProject&&project===$('project').value&&$('evaluation-dialog').open&&id===($('evaluation-dataset').value||evaluationDataset?.id);
  if(!id){panel.replaceChildren(node('p','Сначала выберите сохранённый набор.'));return;}
  try{
    const result=await api('evaluation/datasets/'+encodeURIComponent(id)+'/versions?'+new URLSearchParams({project_id:project,offset:String(offset),limit:'20'}));if(!current())return;
    if(result.id!==id||result.project_id!==project||result.provider_calls!==0||!Array.isArray(result.versions)||result.versions.length>20||result.offset!==offset||result.limit!==20||result.order!=='version_desc'||!Number.isSafeInteger(result.total)||result.total<result.versions.length)throw new Error('Получена несовместимая история набора.');
    if(!Number.isSafeInteger(result.latest_version)||result.latest_version!==result.total||result.total<1||result.total>1000||typeof result.has_more!=='boolean'||result.has_more!==(offset+result.versions.length<result.total)||result.versions.some((v,i)=>v.version!==result.latest_version-offset-i||!Number.isSafeInteger(v.samples)||v.samples<0||typeof v.name!=='string'||typeof v.sha256!=='string'||!/^[a-f0-9]{64}$/.test(v.sha256)))throw new Error('Получены некорректные версии набора.');
    const rows=[node('p','Сохранённых версий: '+result.total)];
    for(const version of result.versions){const row=node('details');row.append(node('summary','Версия '+version.version+' · '+version.name+' · примеров: '+version.samples),node('p','SHA-256: '+version.sha256));const open=node('button','Загрузить версию в редактор'),status=node('p');
      open.onclick=async()=>{if(!current()||open.disabled)return;open.disabled=true;try{
        const snapshot=await api('evaluation/datasets/'+encodeURIComponent(id)+'/versions/'+version.version);if(!current())return;
        if(snapshot.id!==id||snapshot.project_id!==project||snapshot.version!==version.version||snapshot.sha256!==version.sha256)throw new Error('Версия набора отличается от выбранной записи.');
        datasetVersionsEpoch++;showDataset(snapshot);$('evaluation-version').value=snapshot.version;panel.replaceChildren(node('p','Загружена версия '+snapshot.version));
      }catch(error){if(current())status.textContent=error.message;}finally{if(current())open.disabled=false;}};row.append(open,status);rows.push(row);
    }
    const previous=node('button','Предыдущие версии'),next=node('button','Следующие версии');previous.disabled=offset===0;next.disabled=!result.has_more;previous.onclick=()=>{if(current()&&!previous.disabled)return loadDatasetVersions(Math.max(0,offset-20));};next.onclick=()=>{if(current()&&!next.disabled)return loadDatasetVersions(offset+20);};rows.push(previous,next);panel.replaceChildren(...rows);
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}
}
$('dataset-versions-show').onclick=()=>loadDatasetVersions();
let datasetCompareEpoch=0;
function validateDatasetComparisonPage(result,scope,offset){
  if(result.id!==scope.id||result.project_id!==scope.project||result.from?.version!==scope.from||result.to?.version!==scope.to||![result.from.sha256,result.to.sha256].every(h=>typeof h==='string'&&/^[a-f0-9]{64}$/.test(h))||result.provider_calls!==0||result.offset!==offset||result.limit!==100||result.order!=='sample_id_asc'||!Array.isArray(result.changes)||!Number.isSafeInteger(result.total)||result.total<0||result.total>4000||result.changes.length!==Math.min(100,Math.max(0,result.total-offset))||typeof result.has_more!=='boolean'||result.has_more!==(offset+result.changes.length<result.total))throw new Error('Получено несовместимое сравнение версий.');
  if(!result.counts||!['added','removed','changed','unchanged'].every(k=>Number.isSafeInteger(result.counts[k])&&result.counts[k]>=0)||result.counts.added+result.counts.removed+result.counts.changed!==result.total||result.changes.some(c=>!c||!['added','removed','changed'].includes(c.kind)||typeof c.sample_id!=='string'||!/^[a-zA-Z0-9_-]{1,80}$/.test(c.sample_id)||!Array.isArray(c.fields)||new Set(c.fields).size!==c.fields.length||c.fields.some(f=>!['input','expected_output','contexts','metadata'].includes(f))||(c.kind==='changed')!==Boolean(c.fields.length)))throw new Error('Получены некорректные изменения набора.');
}
async function collectDatasetComparisonCsv(reviewed,current){
  const scope={id:reviewed.id,project:reviewed.project_id,from:reviewed.from.version,to:reviewed.to.version};
  const pins=page=>JSON.stringify([page.from.sha256,page.to.sha256,page.total,...['added','removed','changed','unchanged'].map(k=>page.counts[k])]);
  const expected=pins(reviewed),rows=[],counts={added:0,removed:0,changed:0};let previous=null;
  for(let offset=0;offset<=4000;offset+=100){
    if(!current())return null;
    const page=offset===reviewed.offset?reviewed:await api('evaluation/datasets/'+encodeURIComponent(scope.id)+'/compare?'+new URLSearchParams({project_id:scope.project,from_version:String(scope.from),to_version:String(scope.to),offset:String(offset),limit:'100'}));
    if(!current())return null;validateDatasetComparisonPage(page,scope,offset);
    if(pins(page)!==expected)throw new Error('Сравнение изменилось во время выгрузки.');
    for(const change of page.changes){
      if(previous!==null&&change.sample_id<=previous)throw new Error('Нарушен порядок примеров в выгрузке.');previous=change.sample_id;counts[change.kind]++;
      rows.push([scope.id,scope.project,scope.from,page.from.sha256,scope.to,page.to.sha256,change.sample_id,change.kind,change.fields.join(';')]);
    }
    if(!page.has_more){
      if(['added','removed','changed'].some(k=>counts[k]!==page.counts[k]))throw new Error('Выгрузка не совпадает с итогами сравнения.');
      const cell=value=>{let text=String(value);if(/^[=+@-]/.test(text.trimStart())||/^[\t\r\n]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
      return [['dataset_id','project_id','from_version','from_sha256','to_version','to_sha256','sample_id','change','fields'],...rows].map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n';
    }
  }
  throw new Error('Превышен размер выгрузки сравнения.');
}
function downloadDatasetComparisonCsv(csv,reviewed){
  const url=URL.createObjectURL(new Blob(['\ufeff',csv],{type:'text/csv;charset=utf-8'}));const link=node('a');link.href=url;link.download='dataset-'+reviewed.id+'-v'+reviewed.from.version+'-v'+reviewed.to.version+'-changes.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function compareDatasetVersions(offset=0){
  const epoch=++datasetCompareEpoch,project=evaluationProject,id=$('evaluation-dataset').value||evaluationDataset?.id,from=Number($('dataset-compare-from').value),to=Number($('dataset-compare-to').value),panel=$('dataset-compare-result');
  const current=()=>epoch===datasetCompareEpoch&&project===evaluationProject&&project===$('project').value&&$('evaluation-dialog').open&&id===($('evaluation-dataset').value||evaluationDataset?.id)&&from===Number($('dataset-compare-from').value)&&to===Number($('dataset-compare-to').value);
  if(!id){panel.replaceChildren(node('p','Выберите сохранённый набор.'));return;}
  try{
    const result=await api('evaluation/datasets/'+encodeURIComponent(id)+'/compare?'+new URLSearchParams({project_id:project,from_version:String(from),to_version:String(to),offset:String(offset),limit:'100'}));if(!current())return;
    validateDatasetComparisonPage(result,{id,project,from,to},offset);
    const labels={added:'Добавлен',removed:'Удалён',changed:'Изменён'},fields={input:'вопрос',expected_output:'эталонный ответ',contexts:'справочные материалы',metadata:'метаданные'};
    const rows=[node('p','Добавлено: '+result.counts.added+' · удалено: '+result.counts.removed+' · изменено: '+result.counts.changed+' · без изменений: '+result.counts.unchanged),node('p','Название '+(result.name_changed?'изменилось':'не изменилось')),node('p','Версия '+from+' · SHA-256: '+result.from.sha256),node('p','Версия '+to+' · SHA-256: '+result.to.sha256)];
    for(const change of result.changes)rows.push(node('p',labels[change.kind]+' · '+change.sample_id+(change.fields.length?' · '+change.fields.map(f=>fields[f]).join(', '):'')));
    const download=node('button','Скачать все изменения (CSV)'),exportStatus=node('p');
    download.onclick=async()=>{if(!current()||download.disabled)return;download.disabled=true;exportStatus.textContent='Проверяю все страницы…';try{const csv=await collectDatasetComparisonCsv(result,current);if(csv!==null&&current()){downloadDatasetComparisonCsv(csv,result);exportStatus.textContent='Все изменения выгружены.';}}catch(error){if(current())exportStatus.textContent=error.message;}finally{if(current())download.disabled=false;}};
    rows.push(download,exportStatus);
    const previous=node('button','Предыдущие изменения'),next=node('button','Следующие изменения');previous.disabled=offset===0;next.disabled=!result.has_more;previous.onclick=()=>{if(current()&&!previous.disabled)return compareDatasetVersions(Math.max(0,offset-100));};next.onclick=()=>{if(current()&&!next.disabled)return compareDatasetVersions(offset+100);};rows.push(previous,next);panel.replaceChildren(...rows);
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}
}
$('dataset-compare-show').onclick=()=>compareDatasetVersions();
for(const id of ['dataset-compare-from','dataset-compare-to'])$(id).oninput=()=>{datasetCompareEpoch++;$('dataset-compare-result').replaceChildren();};
let selectedDatasetLoadEpoch=0;
async function loadSelectedDataset(){
  const epoch=++selectedDatasetLoadEpoch,origin=evaluationDataset,project=evaluationProject,id=$('evaluation-dataset').value,version=Number($('evaluation-version').value);if(!id)return;
  const current=()=>epoch===selectedDatasetLoadEpoch&&origin===evaluationDataset&&project===evaluationProject&&project===$('project').value&&$('evaluation-dialog').open&&id===$('evaluation-dataset').value&&version===Number($('evaluation-version').value);
  try{
    const snapshot=await api('evaluation/datasets/'+encodeURIComponent(id)+'/versions/'+version);if(!current())return;
    if(snapshot.id!==id||snapshot.project_id!==project||snapshot.version!==version||typeof snapshot.sha256!=='string'||!/^[a-f0-9]{64}$/.test(snapshot.sha256))throw new Error('Получена несовместимая версия набора.');
    showDataset(snapshot);
  }catch(error){if(current())throw error;}
}
$('evaluation-open').onclick=async()=>{try{
  evaluationRunCatalogEpoch++;datasetCompareEpoch++;$('dataset-compare-result').replaceChildren();datasetLifecycleEpoch++;$('dataset-lifecycle-result').replaceChildren();datasetVersionsEpoch++;$('dataset-versions-result').replaceChildren();
  for(const id of ['evaluation-runs-status','evaluation-runs-model','evaluation-runs-kind'])$(id).value='';for(const id of ['evaluation-runs-dataset','evaluation-runs-provider'])$(id).checked=false;
  evaluationProject=$('project').value;showDataset(null);showPrompt(null);evaluationSelectedRun=null;
  clearOfflineScores();$('offline-scores-dataset-only').checked=false;
  evaluationMatrixEpoch++;$('evaluation-matrix-result').replaceChildren();$('evaluation-matrix-import').value='';
  clearJudgePlan();$('judge-plan-id').value='';
  $('evaluation-result').replaceChildren();$('evaluation-comparison').replaceChildren();
  $('evaluation-dialog').showModal();await loadEvaluationDatasets();await loadEvaluationRuns();await loadPrompts();
}catch(error){fail(error);}};
$('evaluation-close').onclick=()=>{evaluationRunCatalogEpoch++;datasetCompareEpoch++;selectedDatasetLoadEpoch++;datasetVersionsEpoch++;datasetLifecycleEpoch++;$('evaluation-dialog').close();};
let reviewedJudgePlan=null,judgePlanStarting=false,judgePlanReviewEpoch=0;
function clearJudgePlan(){judgePlanReviewEpoch++;reviewedJudgePlan=null;$('judge-plan-start').disabled=true;$('judge-plan-preview-result').replaceChildren();$('judge-plan-start-result').textContent='';}
$('judge-plan-id').oninput=clearJudgePlan;
$('judge-plan-preview').onclick=async()=>{try{
  clearJudgePlan();const id=$('judge-plan-id').value.trim(),project=evaluationProject,epoch=judgePlanReviewEpoch;
  const plan=await api('evaluation/judge-plans/'+encodeURIComponent(id));
  if(epoch!==judgePlanReviewEpoch||id!==$('judge-plan-id').value.trim()||project!==evaluationProject)return;
  if(plan.project_id!==project)throw new Error('План принадлежит другому проекту.');
  reviewedJudgePlan=plan;
  $('judge-plan-preview-result').replaceChildren(node('p','Примеров: '+plan.samples.length+' · модель: '+plan.settings.provider+' / '+plan.settings.model+' · версия набора: '+plan.dataset_version),node('p','Критерий оценки'),node('pre',plan.rubric),node('p','Запуск отправит сохранённые вопросы, ответы и справочные материалы выбранному провайдеру. Может потребоваться оплата вызовов модели.'));
  $('judge-plan-start').disabled=judgePlanStarting;
}catch(error){fail(error);}};
$('judge-plan-start').onclick=async()=>{
  const plan=reviewedJudgePlan;if(!plan||judgePlanStarting||plan.project_id!==evaluationProject)return;
  judgePlanStarting=true;$('judge-plan-start').disabled=true;
  try{const run=await api('evaluation/judge-runs',{plan_id:plan.id});
    if(reviewedJudgePlan===plan)$('judge-plan-start-result').textContent='Запуск '+run.id+'. Результаты доступны в «Вызовы и расходы» → «Пакетные оценки проекта».';
    notify('Пакетная оценка запущена: '+run.id);
  }catch(error){fail(error);}finally{judgePlanStarting=false;if(reviewedJudgePlan===plan)$('judge-plan-start').disabled=false;}
};
$('evaluation-new').onclick=()=>{showDataset(null);$('evaluation-dataset').value='';if($('offline-scores-dataset-only').checked)clearOfflineScores();};
$('evaluation-fork').onclick=()=>{
  if(!evaluationDataset){notify('Сначала загрузите сохранённую версию');return;}
  evaluationDatasetOrigin={id:evaluationDataset.id,version:evaluationDataset.version,sha256:evaluationDataset.sha256};
  showDatasetOrigin();evaluationDataset=null;selectedDatasetLoadEpoch++;datasetVersionsEpoch++;$('evaluation-dataset').value='';$('evaluation-version').value=1;$('evaluation-name').value+=' — вариант';$('dataset-versions-result').replaceChildren(node('p','Сохраните вариант как отдельный набор.'));
};
$('evaluation-add').onclick=()=>$('evaluation-samples').append(sampleEditor());
$('evaluation-load').onclick=()=>loadSelectedDataset().catch(fail);
$('evaluation-dataset').onchange=()=>{
  if($('evaluation-runs-dataset').checked){evaluationRunCatalogEpoch++;$('evaluation-runs').replaceChildren();}
  if($('offline-scores-dataset-only').checked)clearOfflineScores();
  const option=$('evaluation-dataset').selectedOptions[0];if(!option?.value){showDataset(null);return;}
  $('evaluation-version').value=option.dataset.version;loadSelectedDataset().catch(fail);
};
$('evaluation-save').onclick=async()=>{try{
  const samples=[...$('evaluation-samples').children].map(row=>({id:row.sampleId,input:row.querySelector('.sample-input').value,expected_output:row.querySelector('.sample-reference').checked?row.querySelector('.sample-expected').value:null,contexts:row.querySelector('.sample-contexts').value===row.originalContextsText?row.originalContexts:row.querySelector('.sample-contexts').value.split(/\n\s*\n/).filter(Boolean),metadata:row.sampleMetadata}));
  const dataset=await api('evaluation/datasets',{id:evaluationDataset?.id,project_id:evaluationProject,name:$('evaluation-name').value,base_version:evaluationDataset?.version||0,samples,origin:evaluationDatasetOrigin});
  showDataset(dataset);await loadEvaluationDatasets(dataset.id);notify('Сохранена версия '+dataset.version);
}catch(error){fail(error);}};
$('evaluation-export').onclick=()=>{
  if(!evaluationDataset){notify('Сначала сохраните набор');return;}
  const url=URL.createObjectURL(new Blob([JSON.stringify(evaluationDataset,null,2)],{type:'application/json'}));
  const link=node('a');link.href=url;link.download='dataset-'+evaluationDataset.id+'-v'+evaluationDataset.version+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('evaluation-import').onchange=async()=>{try{
  const file=$('evaluation-import').files[0];if(!file)return;if(file.size>8*1024*1024)throw new Error('Набор превышает 8 MiB');
  const source=JSON.parse(await file.text());
  const dataset=await api('evaluation/datasets',{project_id:evaluationProject,name:source.name||file.name,base_version:0,samples:source.samples});
  showDataset(dataset);await loadEvaluationDatasets(dataset.id);notify('Импортирован новый набор');
}catch(error){fail(error);}finally{$('evaluation-import').value='';}};
function savedAnswerJudgePanel(source,current){
    const panel=node('details');panel.append(node('summary','Создать план оценки ответов моделью'));
    const choice=node('select'),custom=node('textarea'),message=node('p'),create=node('button','Сохранить план оценки');create.disabled=true;
    custom.placeholder='Критерий и значение баллов от 0 до 1';custom.maxLength=16000;custom.hidden=true;
    const criterionLabel=node('label','Критерий оценки'),customLabel=node('label','Свой критерий');criterionLabel.append(choice);customLabel.append(custom);customLabel.hidden=true;
    panel.append(criterionLabel,customLabel,node('p','План использует выбранную в Studio модель. Сохранение плана не вызывает модель; запуск доступен после просмотра плана.'),create,message);
    let presets=null,loading=false;
    choice.onchange=()=>{custom.hidden=customLabel.hidden=choice.value!=='custom';const preset=presets?.find(preset=>preset.id===choice.value);message.textContent=preset?({input:'Нужен вопрос для каждого примера.',reference:'Нужен эталонный ответ для каждого примера.',contexts_or_reference:'Нужны справочные материалы или эталонный ответ для каждого примера.'}[preset.required_source]||''):'';};
    panel.ontoggle=async()=>{if(!panel.open||presets||loading||!current())return;loading=true;try{
      const result=await api('evaluation/judge-presets');if(!current())return;presets=result.presets;
      choice.replaceChildren(...presets.map(preset=>{const option=node('option',preset.name+' · версия '+preset.version);option.value=preset.id;return option;}));const option=node('option','Свой критерий');option.value='custom';choice.append(option);choice.onchange();create.disabled=false;
    }catch(error){message.textContent=error.message;}finally{loading=false;}};
    create.onclick=async()=>{if(!current())return;create.disabled=true;try{
      const selected=presets?.find(preset=>preset.id===choice.value),judgeSettings={...settings(),project_id:source.project_id,mode:'chat',allow_writes:false};
      if(source.project_id!==evaluationProject||!current())throw new Error('Источник принадлежит другому проекту.');
      const criterion=choice.value==='custom'?{rubric:custom.value}:{judge_preset:{id:selected.id,version:selected.version}};
      const plan=await api('evaluation/judge-plans',{...(source.kind==='offline_score'?{offline_score_id:source.id}:{experiment_id:source.id}),dataset_id:source.dataset_id,dataset_version:source.dataset_version,outputs:Object.fromEntries(source.items.map(item=>[item.sample_id,item.output])),settings:judgeSettings,...criterion});
      if(!current())return;message.textContent='Сохранён план '+plan.id;
      clearJudgePlan();$('judge-plan-id').value=plan.id;$('judge-plan-id').closest('details').open=true;$('judge-plan-preview').click();$('judge-plan-id').scrollIntoView({block:'center'});
    }catch(error){message.textContent=error.message;}finally{create.disabled=false;}};
    return panel;
}
function experimentExportCsv(packet){
  if(packet.kind!=='experiment_export'||packet.schema_version!==1||!Array.isArray(packet.metrics)||!Array.isArray(packet.items))throw new Error('Некорректный экспорт эксперимента');
  const cell=value=>{
    if(value==null)return '""';
    let text=String(value);
    if(typeof value==='string'&&(/^[\t\r\n]/.test(text)||/^[=+@-]/.test(text.trimStart())))text="'"+text;
    return '"'+text.replaceAll('"','""')+'"';
  };
  const headers=['run_id','dataset_id','dataset_version','dataset_sha256','run_status','strict_quality','sample_id','sample_status','duration_ms','has_error','output_truncated','provider','model','trace_id','prompt_id','prompt_version','prompt_sha256',...packet.metrics.map(metric=>'score_'+metric),...(packet.outputs_included?['output']:[])];
  const rows=[headers];
  for(const item of packet.items)rows.push([packet.run_id,packet.dataset_id,packet.dataset_version,packet.dataset_sha256,packet.status,packet.strict_quality,item.sample_id,item.status,item.duration_ms,item.has_error,item.output_truncated,packet.provider,packet.model,packet.trace_id,packet.prompt_ref?.id,packet.prompt_ref?.version,packet.prompt_ref?.sha256,...packet.metrics.map(metric=>item.scores[metric]),...(packet.outputs_included?[item.output]:[])]);
  return rows.map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n';
}
let experimentExportView=null;
function experimentExportPanel(run){
  if(experimentExportView?.id===run.id)return experimentExportView.panel;
  const panel=node('div'),answers=node('input');answers.type='checkbox';answers.checked=false;
  const label=node('label');label.append(answers,' Включить ответы модели');
  const format=node('select');format.setAttribute('aria-label','Формат экспорта результатов');for(const value of ['json','csv']){const option=node('option',value.toUpperCase());option.value=value;format.append(option);}format.value='json';
  const download=node('button','Экспорт результатов');
  download.onclick=async()=>{if(download.disabled)return;download.disabled=true;try{
    const selectedFormat=format.value;
    const packet=await api('evaluation/experiments/'+encodeURIComponent(run.id)+'/export?include_outputs='+String(answers.checked));
    const contents=selectedFormat==='csv'?experimentExportCsv(packet):JSON.stringify(packet,null,2);
    const url=URL.createObjectURL(new Blob([contents],{type:selectedFormat==='csv'?'text/csv;charset=utf-8':'application/json'}));
    const link=node('a');link.href=url;link.download='experiment-'+run.id+'.'+selectedFormat;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(error){fail(error);}finally{download.disabled=false;}};
  panel.append(label,download,format);experimentExportView={id:run.id,panel};return panel;
}
let experimentSamplesView=null;
function experimentSamplesPanel(run){
  if(!experimentSamplesView||experimentSamplesView.id!==run.id){
    const panel=node('div'),search=node('input'),status=node('select'),count=node('p'),body=node('div');
    search.type='search';search.placeholder='Поиск по примеру и ответу';search.setAttribute('aria-label','Поиск результатов эксперимента');
    for(const [value,text] of [['all','Все примеры'],['problems','Ошибки и неполные ответы'],['completed','Завершённые'],['pending','Ожидают'],['running','Выполняются']]){const option=node('option',text);option.value=value;status.append(option);}status.value='all';
    status.setAttribute('aria-label','Статус примеров');panel.append(search,status,count,body);
    const sort=node('select');sort.setAttribute('aria-label','Порядок примеров');
    for(const [value,text] of [['original','Порядок набора'],['slowest','Сначала медленные'],['fastest','Сначала быстрые'],...(run.metrics||[]).map(metric=>['score:'+metric,'Сначала низкие оценки: '+(metricLabels[metric]||metric)])]){const option=node('option',text);option.value=value;sort.append(option);}sort.value='original';panel.append(sort);
    experimentSamplesView={id:run.id,panel,search,status,count,body,sort,items:[],opened:new Set()};
    const view=experimentSamplesView;
    const update=()=>renderExperimentSamples(view);
    search.oninput=update;status.onchange=update;sort.onchange=update;
  }
  experimentSamplesView.items=run.items;
  renderExperimentSamples(experimentSamplesView);
  return experimentSamplesView.panel;
}
function renderExperimentSamples(view){
  const query=view.search.value.trim().toLocaleLowerCase();
  const items=view.items.filter(item=>{
    const status=view.status.value;
    if(status==='problems'&&!item.error&&!item.output_truncated&&!['failed','cancelled','interrupted','timed_out'].includes(item.status))return false;
    if(status!=='all'&&status!=='problems'&&item.status!==status)return false;
    return !query||[item.sample_id,item.output,item.error].some(value=>typeof value==='string'&&value.toLocaleLowerCase().includes(query));
  });
  const sort=view.sort.value;
  if(sort!=='original'){
    const value=item=>sort.startsWith('score:')?item.scores[sort.slice(6)]:item.duration_ms;
    items.sort((a,b)=>{const av=value(a),bv=value(b),ak=typeof av==='number'&&Number.isFinite(av),bk=typeof bv==='number'&&Number.isFinite(bv);
      if(ak!==bk)return ak?-1:1;if(!ak)return 0;return sort==='slowest'?bv-av:av-bv;});
  }
  view.count.textContent='Показано '+items.length+' из '+view.items.length;
  const rows=[];
  for(const item of items){
    const detail=node('details');detail.dataset.sampleId=item.sample_id;detail.open=view.opened.has(item.sample_id);
    detail.ontoggle=()=>{if(detail.open)view.opened.add(item.sample_id);else view.opened.delete(item.sample_id);};
    detail.append(node('summary',item.sample_id+' · '+(evaluationLabels[item.status]||item.status)));
    if(Number.isSafeInteger(item.duration_ms)&&item.duration_ms>=0)detail.append(node('p','Длительность: '+(item.duration_ms/1000).toFixed(2)+' с'));
    if(item.output!=null)detail.append(node('pre',item.output));
    for(const [metric,score] of Object.entries(item.scores))detail.append(node('p',(metricLabels[metric]||metric)+': '+score));
    if(item.error)detail.append(node('p',({provider_error:'Ошибка провайдера',timeout:'Истекло время ожидания',incomplete_or_oversize_output:'Ответ неполный или превышает лимит',process_restart:'Проверка прервана перезапуском Studio'})[item.error]||item.error));
    rows.push(detail);
  }
  if(!rows.length)rows.push(node('p','Нет примеров, соответствующих фильтру'));
  view.body.replaceChildren(...rows);
}
function playgroundSavedSamplePanel(run,current){
  const panel=node('div'),open=node('button','Вопрос и эталон этого запуска'),body=node('div');panel.append(open,body);
  open.onclick=async()=>{if(!current()||open.disabled)return;open.disabled=true;try{
    const snapshot=await api('evaluation/datasets/'+encodeURIComponent(run.dataset_id)+'/versions/'+run.dataset_version);if(!current())return;
    if(snapshot.id!==run.dataset_id||snapshot.project_id!==run.project_id||snapshot.version!==run.dataset_version||snapshot.sha256!==run.dataset_sha256||!Array.isArray(snapshot.samples)||snapshot.samples.length!==1||snapshot.samples[0].id!==run.items[0]?.sample_id)throw new Error('Сохранённый пример не совпадает с запуском.');
    const sample=snapshot.samples[0];if(typeof sample.input!=='string'||(sample.expected_output!==null&&typeof sample.expected_output!=='string')||!Array.isArray(sample.contexts)||sample.contexts.some(c=>typeof c!=='string'))throw new Error('Получен некорректный сохранённый пример.');const rows=[node('p','Набор · версия '+snapshot.version+' · SHA-256: '+snapshot.sha256),node('p','Вопрос'),node('pre',sample.input),node('p',sample.expected_output===null?'Эталонный ответ не задан':'Эталонный ответ')];if(sample.expected_output!==null)rows.push(node('pre',sample.expected_output));if(sample.contexts.length)rows.push(node('p','Справочные материалы'),node('pre',sample.contexts.join('\n\n')));body.replaceChildren(...rows);
  }catch(error){if(current())body.replaceChildren(node('p',error.message));}finally{open.disabled=false;}};
  return panel;
}
function showEvaluationRun(run){
  evaluationRunStatus=run.status;
  const rows=[node('h3',run.settings.model+' · '+(evaluationLabels[run.status]||run.status))];
  if(run.playground){rows.push(node('p',run.metrics.length?'Пробный запуск с выбранными метриками.':'Пробный запуск без оценки качества.'),playgroundSavedSamplePanel(run,()=>evaluationSelectedRun===run.id&&run.project_id===evaluationProject&&$('evaluation-dialog').open));}
  rows.push(experimentExportPanel(run));
  if(run.prompt_snapshot)rows.push(node('p',run.prompt_snapshot.name+' · версия '+run.prompt_snapshot.version+' · '+run.prompt_snapshot.sha256.slice(0,12)));
  if(run.status==='running'){
    const cancel=node('button','Отменить проверку');cancel.onclick=async()=>{try{await api('evaluation/experiments/'+run.id+'/cancel',{});}catch(error){fail(error);}};rows.push(cancel);
  }
  if(run.status==='completed'&&run.items.length&&run.items.every(item=>item.status==='completed'&&!item.output_truncated&&typeof item.output==='string')){
    rows.push(savedAnswerJudgePanel(run,()=>run.project_id===evaluationProject&&evaluationSelectedRun===run.id&&$('evaluation-dialog').open));
  }
  rows.push(experimentSamplesPanel(run));
  if(run.trace_id){
    const tracePanel=node('div');tracePanel.className='experiment-trace';
    const open=node('button','Вызовы проверки');open.onclick=async()=>{open.disabled=true;try{const result=await api('observability/traces?session_id='+encodeURIComponent('experiment-'+run.id));const trace=result.traces.find(trace=>trace.id===run.trace_id);if(!trace)throw new Error('Журнал проверки не найден');const view=traceView(trace);view.open=true;tracePanel.replaceChildren(view);}catch(error){open.disabled=false;fail(error);}};
    tracePanel.append(open);rows.push(tracePanel);
  }
  if(run.recovered_ms)rows.unshift(node('p','Проверка прервана завершением Studio. Готовые результаты сохранены; оставшиеся примеры автоматически не запускались.'));
  $('evaluation-result').replaceChildren(...rows);
}
let evaluationRunCatalogEpoch=0;
function evaluationRunFilters(){
  const filters={status:$('evaluation-runs-status').value,model:$('evaluation-runs-model').value,playground:$('evaluation-runs-kind').value,dataset_id:$('evaluation-runs-dataset').checked?($('evaluation-dataset').value||evaluationDataset?.id||''):'',provider:$('evaluation-runs-provider').checked?$('provider').value:''};
  return filters;
}
async function loadEvaluationRuns(offset=0){
  const project=evaluationProject,epoch=++evaluationRunCatalogEpoch,filters=evaluationRunFilters(),signature=JSON.stringify(filters);
  if($('evaluation-runs-dataset').checked&&!filters.dataset_id)throw new Error('Сначала выберите набор для поиска запусков.');
  const current=()=>epoch===evaluationRunCatalogEpoch&&project===evaluationProject&&project===$('project').value&&$('evaluation-dialog').open&&signature===JSON.stringify(evaluationRunFilters());
  const query=new URLSearchParams({project_id:project,offset:String(offset),limit:'20'});for(const [key,value]of Object.entries(filters))if(value!=='')query.set(key,value);
  let result,choices;try{[result,choices]=await Promise.all([api('evaluation/experiments?'+query),api('evaluation/experiments?project_id='+encodeURIComponent(project))]);}catch(error){if(current())throw error;return;}if(!current())return;
  if(result.offset!==offset||result.limit!==20||result.order!=='id_desc'||result.provider_calls!==0||!Number.isSafeInteger(result.total)||result.total<0||result.total>1000||!Array.isArray(result.runs)||result.runs.length!==Math.min(20,Math.max(0,result.total-offset))||typeof result.has_more!=='boolean'||result.has_more!==(offset+result.runs.length<result.total)||result.runs.some(run=>run.project_id!==project)||!Array.isArray(choices.runs)||choices.runs.length>1000||choices.runs.some(run=>run.project_id!==project))throw new Error('Получен несовместимый каталог запусков.');
  const valid=run=>typeof run.id==='string'&&/^[a-f0-9-]{1,80}$/.test(run.id)&&typeof run.dataset_sha256==='string'&&/^[a-f0-9]{64}$/.test(run.dataset_sha256)&&typeof run.model==='string'&&typeof run.provider==='string'&&typeof run.strict_quality==='boolean'&&typeof run.playground==='boolean'&&['running','completed','failed','cancelled','interrupted'].includes(run.status);
  if([...result.runs,...choices.runs].some(run=>!valid(run))||result.runs.some(run=>filters.status&&run.status!==filters.status||filters.provider&&run.provider!==filters.provider||filters.dataset_id&&run.dataset_id!==filters.dataset_id||filters.playground!==''&&run.playground!==(filters.playground==='true')||filters.model.trim()&&!run.model.toLowerCase().includes(filters.model.trim().toLowerCase()))||result.runs.some((run,i)=>i>0&&run.id>=result.runs[i-1].id))throw new Error('Запуски не соответствуют выбранным фильтрам.');
  const rows=[node('p','Найдено запусков: '+result.total)];
  for(const run of result.runs){
    const row=node('div'),open=node('button','Открыть');open.onclick=async()=>{if(!current()||open.disabled)return;open.disabled=true;try{evaluationSelectedRun=run.id;const detail=await api('evaluation/experiments/'+run.id);if(current()&&evaluationSelectedRun===run.id){if(detail.id!==run.id||detail.project_id!==project||detail.dataset_sha256!==run.dataset_sha256)throw new Error('Запуск не совпадает с выбранной записью.');showEvaluationRun(detail);}}catch(error){if(current())fail(error);}finally{if(current())open.disabled=false;}};
    row.append(node('span',run.model+' · '+(evaluationLabels[run.status]||run.status)+(run.playground?' · Playground':'')+' · '+run.id.slice(-10)),open);rows.push(row);
  }
  const previous=node('button','Предыдущие запуски'),next=node('button','Следующие запуски');previous.disabled=offset===0;next.disabled=!result.has_more;previous.onclick=()=>{if(current()&&!previous.disabled)return loadEvaluationRuns(Math.max(0,offset-20)).catch(fail);};next.onclick=()=>{if(current()&&!next.disabled)return loadEvaluationRuns(offset+20).catch(fail);};rows.push(previous,next);$('evaluation-runs').replaceChildren(...rows);
  for(const name of ['evaluation-baseline','evaluation-candidate']){
    const selected=$(name).value;$(name).replaceChildren(...choices.runs.filter(r=>r.strict_quality).map(run=>{const option=node('option',run.model+' · '+run.id.slice(-10));option.value=run.id;return option;}));
    if([...$(name).options].some(o=>o.value===selected))$(name).value=selected;
  }
}
for(const id of ['evaluation-runs-status','evaluation-runs-model','evaluation-runs-kind','evaluation-runs-dataset','evaluation-runs-provider'])$(id).oninput=()=>{evaluationRunCatalogEpoch++;$('evaluation-runs').replaceChildren();};
$('evaluation-refresh').onclick=()=>loadEvaluationRuns().catch(fail);
$('evaluation-run').onclick=async()=>{try{
  if(!evaluationDataset)throw new Error('Сначала сохраните набор');
  if(promptOrigin&&!evaluationPrompt)throw new Error('Сначала сохраните вариант задания');
  const metrics=[];if($('evaluation-bigram-f1').checked)metrics.push('character_bigram_f1');if($('evaluation-exact').checked)metrics.push('exact_match');if($('evaluation-contains').checked)metrics.push('contains_reference');if($('evaluation-json').checked)metrics.push('json_valid');if($('evaluation-json-equals').checked)metrics.push('json_equals');if($('evaluation-token-f1').checked)metrics.push('whitespace_token_f1');
  const run=await api('evaluation/experiments',{dataset_id:evaluationDataset.id,dataset_version:evaluationDataset.version,settings:{...settings(),project_id:evaluationProject,mode:'chat',allow_writes:false},...(evaluationPrompt?{prompt_ref:{id:evaluationPrompt.id,version:evaluationPrompt.version}}:{prompt_template:$('evaluation-prompt').value}),metrics});
  evaluationSelectedRun=run.id;showEvaluationRun(run);await loadEvaluationRuns();
}catch(error){fail(error);}};
function comparisonExportCsv(receipt){
  const headers=['comparison_id','baseline_id','candidate_id','project_id','dataset_id','dataset_version','dataset_sha256','eligible','regressions','improvements','reason','sample_id','metric','baseline','candidate','delta','change'];
  const cell=value=>{if(value==null)return '""';let text=String(value);if(typeof value==='string'&&(/^[\t\r\n]/.test(text)||/^[=+@-]/.test(text.trimStart())))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
  const fixed=[receipt.id,receipt.baseline_id,receipt.candidate_id,receipt.project_id,receipt.dataset_id,receipt.dataset_version,receipt.dataset_sha256,receipt.eligible,receipt.regressions,receipt.improvements,receipt.reason];
  return [headers,...receipt.pairs.map(pair=>[...fixed,pair.sample_id,pair.metric,pair.baseline,pair.candidate,pair.delta,pair.delta<0?'regression':pair.delta>0?'improvement':'unchanged'])].map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n';
}
function experimentComparisonPanel(receipt){
  const panel=node('div'),filter=node('select'),pairs=node('div'),count=node('p');
  panel.append(node('h4',receipt.eligible?'Улучшение подтверждено':'Улучшение не подтверждено'),node('p','Ухудшений: '+receipt.regressions+' · улучшений: '+receipt.improvements),node('p','Сравнение не меняет настройки модели.'));
  for(const [value,text] of [['all','Все оценки'],['regressions','Ухудшения'],['improvements','Улучшения'],['unchanged','Без изменений']]){const option=node('option',text);option.value=value;filter.append(option);}filter.value=receipt.regressions?'regressions':'all';filter.setAttribute('aria-label','Изменение оценки');
  let answerRequest=null;
  const answers=()=>answerRequest||(answerRequest=Promise.all([api('evaluation/experiments/'+encodeURIComponent(receipt.baseline_id)),api('evaluation/experiments/'+encodeURIComponent(receipt.candidate_id))]).catch(error=>{answerRequest=null;throw error;}));
  const render=()=>{
    const selected=receipt.pairs.filter(pair=>filter.value==='all'||filter.value==='regressions'&&pair.delta<0||filter.value==='improvements'&&pair.delta>0||filter.value==='unchanged'&&pair.delta===0);
    count.textContent='Показано '+selected.length+' из '+receipt.pairs.length;
    const rows=[];
    for(const pair of selected){
      const row=node('details'),body=node('div');row.append(node('summary',pair.sample_id+' · '+(metricLabels[pair.metric]||pair.metric)+': '+pair.baseline+' → '+pair.candidate));
      const open=node('button','Сравнить ответы');open.onclick=async()=>{open.disabled=true;try{
        const [base,next]=await answers();
        if(base.id!==receipt.baseline_id||next.id!==receipt.candidate_id||['project_id','dataset_id','dataset_version','dataset_sha256'].some(key=>base[key]!==receipt[key]||next[key]!==receipt[key]))throw new Error('Ответы не соответствуют сравнению');
        const a=base.items.find(item=>item.sample_id===pair.sample_id),b=next.items.find(item=>item.sample_id===pair.sample_id);
        if(!a||!b||typeof a.output!=='string'||typeof b.output!=='string')throw new Error('Ответ примера недоступен');
        body.replaceChildren(node('h5','Базовый ответ'),node('pre',a.output),node('h5','Новый ответ'),node('pre',b.output));
      }catch(error){body.replaceChildren(node('p',error.message));}finally{open.disabled=false;}};
      row.append(open,body);rows.push(row);
    }
    if(!rows.length)rows.push(node('p','Нет оценок, соответствующих фильтру'));
    pairs.replaceChildren(...rows);
  };
  const format=node('select');format.setAttribute('aria-label','Формат экспорта сравнения');for(const value of ['json','csv']){const option=node('option',value.toUpperCase());option.value=value;format.append(option);}format.value='json';
  const download=node('button','Экспорт сравнения');download.onclick=async()=>{if(download.disabled)return;download.disabled=true;try{
    const selectedFormat=format.value;
    const verified=await api('evaluation/comparisons/'+encodeURIComponent(receipt.id));
    if(verified.id!==receipt.id||verified.baseline_id!==receipt.baseline_id||verified.candidate_id!==receipt.candidate_id)throw new Error('Сравнение не соответствует выбранному результату');
    const contents=selectedFormat==='csv'?comparisonExportCsv(verified):JSON.stringify(verified,null,2);
    const url=URL.createObjectURL(new Blob([contents],{type:selectedFormat==='csv'?'text/csv;charset=utf-8':'application/json'})),link=node('a');
    link.href=url;link.download='comparison-'+receipt.id+'.'+selectedFormat;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(error){fail(error);}finally{download.disabled=false;}};
  filter.onchange=render;panel.append(filter,count,pairs,download,format);render();return panel;
}
$('evaluation-compare').onclick=async()=>{try{
  const baselineId=$('evaluation-baseline').value,candidateId=$('evaluation-candidate').value,project=evaluationProject;
  const receipt=await api('evaluation/compare',{baseline_id:baselineId,candidate_id:candidateId});
  if(project!==evaluationProject||baselineId!==$('evaluation-baseline').value||candidateId!==$('evaluation-candidate').value)return;
  $('evaluation-comparison-id').value=receipt.id;
  $('evaluation-comparison').replaceChildren(experimentComparisonPanel(receipt));
}catch(error){fail(error);}};
let comparisonCatalogRequest=0;
$('evaluation-comparisons-show').onclick=()=>{
  const project=evaluationProject;
  const show=async offset=>{const version=++comparisonCatalogRequest;try{
    const result=await api('evaluation/comparisons?'+new URLSearchParams({project_id:project,offset:String(offset),limit:'30'}));
    if(project!==evaluationProject||version!==comparisonCatalogRequest)return;
    const rows=[node('p','Сохранённые сравнения. Результат проверяется при открытии.')];
    for(const receipt of result.comparisons){
      const row=node('div'),open=node('button','Открыть');
      open.onclick=()=>{if(project!==evaluationProject)return;$('evaluation-comparison-id').value=receipt.id;$('evaluation-comparison-open').click();};
      row.append(node('span',receipt.dataset_id+' · версия '+receipt.dataset_version+' · '+receipt.id),open);rows.push(row);
    }
    if(!result.comparisons.length)rows.push(node('p','Нет сохранённых сравнений'));
    if(offset>0){const previous=node('button','Назад');previous.onclick=()=>show(Math.max(0,offset-30));rows.push(previous);}
    if(result.has_more){const next=node('button','Далее');next.onclick=()=>show(offset+30);rows.push(next);}
    $('evaluation-comparisons-list').replaceChildren(...rows);
  }catch(error){if(project===evaluationProject&&version===comparisonCatalogRequest)fail(error);}};
  return show(0);
};
$('evaluation-comparison-open').onclick=async()=>{const button=$('evaluation-comparison-open');if(button.disabled)return;button.disabled=true;try{
  const id=$('evaluation-comparison-id').value.trim(),project=evaluationProject;
  if(!/^[A-Za-z0-9-]{1,80}$/.test(id))throw new Error('Укажите идентификатор сравнения');
  const receipt=await api('evaluation/comparisons/'+encodeURIComponent(id));
  if(project!==evaluationProject||id!==$('evaluation-comparison-id').value.trim())return;
  if(receipt.id!==id)throw new Error('Идентификатор сравнения не совпадает');
  if(receipt.project_id!==project)throw new Error('Сравнение относится к другому проекту');
  $('evaluation-comparison').replaceChildren(experimentComparisonPanel(receipt));
}catch(error){fail(error);}finally{button.disabled=false;}};
let evaluationPolling=false;
setInterval(async()=>{
  if(!$('evaluation-dialog').open||!evaluationSelectedRun||evaluationRunStatus!=='running'||evaluationPolling)return;
  evaluationPolling=true;const id=evaluationSelectedRun;
  try{const run=await api('evaluation/experiments/'+id);if(evaluationSelectedRun===id){showEvaluationRun(run);if(run.status!=='running')await loadEvaluationRuns();}}catch(error){fail(error);}finally{evaluationPolling=false;}
},1200);
let traceSession = null;
function reviewQueueCreationPanel(trace){
  const panel=node('details'),name=node('input'),instructions=node('textarea'),target=node('select'),save=node('button','Сохранить очередь'),message=node('p');
  panel.append(node('summary','Создать очередь проверки для этого вызова'));name.setAttribute('aria-label','Название очереди');name.maxLength=200;instructions.setAttribute('aria-label','Инструкции проверяющему');instructions.maxLength=16000;
  target.setAttribute('aria-label','Источники проверки');target.multiple=true;target.size=Math.min(8,trace.spans.length+1);target.append(new Option('Весь вызов',''));for(const span of trace.spans)target.append(new Option(span.name+' · шаг '+span.id,String(span.id)));
  panel.append(node('p','Название очереди'),name,node('p','Инструкции проверяющему'),instructions,node('p','Выберите весь вызов или несколько шагов (Ctrl / Cmd для нескольких источников).'),target,save,message);
  const current=()=>trace.project_id===$('project').value&&panel.isConnected!==false;
  save.onclick=async()=>{if(!current()||save.disabled)return;save.disabled=true;const selected=target.selectedOptions?Array.from(target.selectedOptions,option=>option.value):[target.value];const body={project_id:trace.project_id,name:name.value,instructions:instructions.value,targets:selected.map(value=>({trace_id:trace.id,span_id:value===''?null:Number(value)}))};
    try{if(!body.targets.length||body.targets.length>200)throw new Error('Выберите от 1 до 200 источников');if(new TextEncoder().encode(body.name).length>200||new TextEncoder().encode(body.instructions).length>16000)throw new Error('Название или инструкции слишком длинные');const queue=await api('observability/review-queues',body);if(!current())return;if(typeof queue.id!=='string'||!queue.id||queue.project_id!==body.project_id||queue.name!==body.name||queue.instructions!==body.instructions||queue.version!==1||JSON.stringify(queue.targets)!==JSON.stringify(body.targets))throw new Error('Источник сохранённой очереди не совпадает');message.textContent='Очередь сохранена. Откройте «Очереди ручной проверки» в контексте проекта.';}catch(error){if(current())message.textContent=error.message;}finally{save.disabled=false;}};
  return panel;
}
function reviewQueueHistoryPanel(queue,current){
  const panel=node('details'),load=node('button','Показать изменения'),body=node('div');panel.append(node('summary','История очереди'),load,body);let epoch=0;
  const actions={create:'Создана очередь',assignment:'Изменено назначение',complete:'Пункт завершён',reopen:'Пункт открыт повторно',archive:'Очередь в архиве',restore:'Очередь восстановлена'};
  async function page(offset){
    if(!current()||load.disabled)return;load.disabled=true;const token=++epoch,active=()=>current()&&token===epoch;
    try{const packet=await api('observability/review-queues/'+encodeURIComponent(queue.id)+'/history?offset='+offset+'&limit=20');if(!active())return;
      if(packet.kind!=='review_queue_history'||packet.queue_id!==queue.id||packet.project_id!==queue.project_id||packet.queue_version!==queue.version||packet.offset!==offset||packet.limit!==20||packet.order!=='version_desc'||packet.provider_calls!==0||typeof packet.history_complete!=='boolean'||!Number.isInteger(packet.total)||packet.total<0||packet.total>2000||!Array.isArray(packet.entries)||packet.entries.length!==Math.min(20,Math.max(0,packet.total-offset))||packet.has_more!==(offset+packet.entries.length<packet.total))throw new Error('История не совпадает с показанной очередью. Откройте очередь заново.');
      for(let i=0;i<packet.entries.length;i++){const entry=packet.entries[i];if(!Number.isSafeInteger(entry.version)||entry.version<1||entry.version>queue.version||i>0&&packet.entries[i-1].version<=entry.version||!Object.hasOwn(actions,entry.action)||typeof entry.archived!=='boolean'||entry.target_index!=null&&(!Number.isInteger(entry.target_index)||entry.target_index<0||entry.target_index>=queue.targets.length)||entry.reviewer!=null&&typeof entry.reviewer!=='string'||entry.action==='complete'&&(!entry.reviewer||!Number.isInteger(entry.feedback_version)||entry.feedback_version<1||entry.feedback_version>2000||typeof entry.annotation_id!=='string'||!entry.annotation_id))throw new Error('Некорректная запись истории');}
      body.replaceChildren(node('p','Записей: '+packet.total),node('p',packet.history_complete?'История с момента создания очереди.':'История неполная: более ранние изменения не записывались.'));
      for(const entry of packet.entries){const row=node('div');row.append(node('p','Версия '+entry.version+' · '+actions[entry.action]+(entry.target_index==null?'':' · пункт '+(entry.target_index+1))));if(entry.action==='assignment')row.append(node('p',entry.reviewer?'Проверяющий: '+entry.reviewer:'Назначение снято'));if(entry.action==='complete')row.append(node('p','Проверяющий: '+entry.reviewer+' · версия отзывов '+entry.feedback_version),node('p','Сохранённый отзыв: '+entry.annotation_id));body.append(row);}
      const prev=node('button','Предыдущие изменения'),next=node('button','Следующие изменения');prev.disabled=offset===0;next.disabled=!packet.has_more;prev.onclick=()=>{if(active())return page(Math.max(0,offset-20));};next.onclick=()=>{if(active())return page(offset+20);};body.append(prev,next);
    }catch(error){if(active())body.replaceChildren(node('p',error.message));}finally{load.disabled=false;}
  }
  load.onclick=()=>page(0);panel.ontoggle=()=>{if(!panel.open)epoch++;};return panel;
}
function reviewQueueExportCsv(queue){
  if(typeof queue.id!=='string'||!queue.id||typeof queue.project_id!=='string'||typeof queue.name!=='string'||!Number.isSafeInteger(queue.version)||queue.version<1||!Array.isArray(queue.targets)||!queue.targets.length||queue.targets.length>200||!queue.assignments||!queue.completions)throw new Error('Некорректная очередь для экспорта');
  const keys=new Set(queue.targets.map((_,i)=>String(i))),seen=new Set();if([...Object.keys(queue.assignments),...Object.keys(queue.completions)].some(key=>!keys.has(key)))throw new Error('Некорректные назначения');
  const rows=[['queue_id','queue_version','project_id','queue_name','archived','target_index','trace_id','span_id','status','reviewer','feedback_version','annotation_id']];
  queue.targets.forEach((target,index)=>{const reviewer=queue.assignments[index],pin=queue.completions[index],identity=JSON.stringify([target.trace_id,target.span_id??null]);if(typeof target.trace_id!=='string'||!target.trace_id||target.span_id!=null&&(!Number.isSafeInteger(target.span_id)||target.span_id<0)||seen.has(identity)||reviewer!=null&&(typeof reviewer!=='string'||!reviewer.trim())||pin&&(pin.reviewer!==reviewer||!reviewer||!Number.isInteger(pin.feedback_version)||pin.feedback_version<1||pin.feedback_version>2000||typeof pin.annotation_id!=='string'||!pin.annotation_id))throw new Error('Некорректный источник или отзыв');seen.add(identity);rows.push([queue.id,queue.version,queue.project_id,queue.name,queue.archived?'true':'false',index,target.trace_id,target.span_id,pin?'completed':reviewer?'assigned':'unassigned',reviewer,pin?.feedback_version,pin?.annotation_id]);});
  const cell=value=>{let text=value==null?'':String(value);if(typeof value==='string'&&(/^[\t\r\n]/.test(text)||/^[=+@-]/.test(text.trimStart())))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};return rows.map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n';
}
function reviewQueueCompletionPanel(queue,index,current,onUpdated){
  const panel=node('div'),target=queue.targets[index],reviewer=queue.assignments[index],evidence=queue.completions[index],message=node('p');
  if(queue.archived){panel.append(node('p','Очередь в архиве. Восстановите её для изменения пунктов.'));return panel;}
  const accept=updated=>{if(updated.id!==queue.id||updated.project_id!==queue.project_id||updated.version!==queue.version+1||JSON.stringify(updated.targets)!==JSON.stringify(queue.targets))throw new Error('Изменённая очередь не совпадает');onUpdated(updated);};
  if(evidence){
    panel.append(node('p','Основание проверки: '+evidence.reviewer+' · версия отзывов '+evidence.feedback_version));const reopen=node('button','Открыть пункт повторно');panel.append(reopen,message);
    reopen.onclick=async()=>{if(!current()||reopen.disabled)return;reopen.disabled=true;try{const updated=await api('observability/review-queues/'+encodeURIComponent(queue.id)+'/completion',{base_version:queue.version,target_index:index,action:'reopen'});if(!current())return;if(updated.completions?.[index])throw new Error('Пункт не открыт повторно');accept(updated);}catch(error){if(current())message.textContent=error.message;}finally{reopen.disabled=false;}};return panel;
  }
  if(!reviewer){panel.append(node('p','Для завершения сначала назначьте проверяющего.'));return panel;}
  const version=node('input'),load=node('button','Выбрать сохранённый отзыв'),choices=node('select'),preview=node('div'),finish=node('button','Завершить пункт с этим отзывом');version.type='number';version.min=1;version.max=2000;version.setAttribute('aria-label','Версия отзывов для завершения');choices.setAttribute('aria-label','Отзыв для завершения');finish.disabled=true;panel.append(node('p','Версия отзывов (пусто — текущая)'),version,load,choices,preview,finish,message);let selection=0,packet=null,annotations=[];
  const clear=()=>{selection++;packet=null;annotations=[];choices.replaceChildren();preview.replaceChildren();finish.disabled=true;};version.oninput=clear;
  function show(){const annotation=annotations.find(a=>a.id===choices.value);preview.replaceChildren();finish.disabled=!annotation;if(annotation){preview.append(node('p',annotation.author+' · версия '+packet.version));if(annotation.metric)preview.append(node('p',annotation.metric+': '+(annotation.value??annotation.category)));if(annotation.comment)preview.append(node('pre',annotation.comment));if(annotation.correction)preview.append(node('p','Исправленный ответ'),node('pre',annotation.correction));}}
  choices.onchange=show;
  load.onclick=async()=>{if(!current()||load.disabled)return;clear();const epoch=selection,selectedVersion=version.value;load.disabled=true;
    try{if(selectedVersion!==''&&(!Number.isInteger(Number(selectedVersion))||Number(selectedVersion)<1||Number(selectedVersion)>2000))throw new Error('Выберите версию отзывов от 1 до 2000');const receipt=await api('observability/traces/'+encodeURIComponent(target.trace_id)+'/feedback'+(selectedVersion===''?'':'?version='+Number(selectedVersion)));if(!current()||epoch!==selection||selectedVersion!==version.value)return;if(receipt.trace_id!==target.trace_id||!Number.isInteger(receipt.version)||receipt.version<0||receipt.version>2000||selectedVersion!==''&&receipt.version!==Number(selectedVersion)||!Array.isArray(receipt.annotations))throw new Error('Версия отзывов не совпадает');packet=receipt;annotations=receipt.annotations.filter(a=>typeof a.id==='string'&&a.author===reviewer&&(a.span_id??null)===(target.span_id??null)&&a.deleted===false);if(receipt.version===0)annotations=[];for(const annotation of annotations)choices.append(new Option(annotation.author+' · '+(annotation.metric||'Комментарий')+(annotation.value!=null?' · '+annotation.value:annotation.category?' · '+annotation.category:''),annotation.id));choices.value=annotations[0]?.id||'';show();message.textContent=annotations.length?'Выберите отзыв и проверьте его перед завершением.':'В этой версии нет действующего отзыва назначенного проверяющего для этого источника.';}catch(error){if(current()&&epoch===selection)message.textContent=error.message;}finally{load.disabled=false;}};
  finish.onclick=async()=>{if(!current()||finish.disabled)return;const annotation=annotations.find(a=>a.id===choices.value);if(!annotation||!packet)return;const body={base_version:queue.version,target_index:index,action:'complete',feedback_version:packet.version,annotation_id:annotation.id},epoch=selection;finish.disabled=true;
    try{const updated=await api('observability/review-queues/'+encodeURIComponent(queue.id)+'/completion',body);if(!current()||epoch!==selection)return;const pin=updated.completions?.[index];if(pin?.reviewer!==reviewer||pin?.feedback_version!==body.feedback_version||pin?.annotation_id!==body.annotation_id)throw new Error('Основание завершения не совпадает');accept(updated);}catch(error){if(current()&&epoch===selection)message.textContent=error.message;}finally{if(current()&&epoch===selection)finish.disabled=false;}};
  return panel;
}
let reviewQueueEpoch=0,reviewQueueViewEpoch=0,reviewQueueProject=null;
const reviewQueueCurrent=(epoch,project)=>epoch===reviewQueueEpoch&&project===reviewQueueProject&&project===$('project').value&&$('review-queues-dialog').open;
function showReviewQueue(queue){
  const project=reviewQueueProject,epoch=reviewQueueEpoch,view=++reviewQueueViewEpoch,current=()=>reviewQueueCurrent(epoch,project)&&view===reviewQueueViewEpoch,body=$('review-queue-detail');
  if(queue.project_id!==project||typeof queue.id!=='string'||!Number.isInteger(queue.version)||queue.version<1||!Array.isArray(queue.targets)||queue.targets.length<1||queue.targets.length>200||!queue.assignments||!queue.completions)throw new Error('Некорректная очередь проверки');
  body.replaceChildren(node('h3',queue.name),node('pre',queue.instructions));let busy=false;
  queue.targets.forEach((target,index)=>{
    const row=node('div'),reviewer=node('input'),save=node('button','Назначить проверяющего'),source=node('button','Открыть источник'),message=node('p'),content=node('div');reviewer.value=queue.assignments[index]||'';reviewer.maxLength=200;reviewer.setAttribute('aria-label','Проверяющий пункта '+(index+1));save.disabled=queue.archived||!!queue.completions[index];reviewer.disabled=save.disabled;
    row.append(node('p','Пункт '+(index+1)+' · '+(queue.completions[index]?'Проверен':queue.assignments[index]?'Назначен':'Ожидает назначения')),reviewer,save,source,message,content,reviewQueueCompletionPanel(queue,index,current,showReviewQueue));body.append(row);
    save.onclick=async()=>{if(!current()||busy||save.disabled)return;busy=true;save.disabled=true;try{const updated=await api('observability/review-queues/'+encodeURIComponent(queue.id)+'/assignments',{base_version:queue.version,target_index:index,reviewer:reviewer.value||null});if(!current())return;if(updated.id!==queue.id||updated.project_id!==project||updated.version!==queue.version+1||JSON.stringify(updated.targets)!==JSON.stringify(queue.targets))throw new Error('Изменённая очередь не совпадает');showReviewQueue(updated);}catch(error){if(current())message.textContent=error.message;}finally{busy=false;save.disabled=queue.archived||!!queue.completions[index];}};
    source.onclick=async()=>{if(!current()||source.disabled)return;source.disabled=true;try{const trace=await api('observability/traces/'+encodeURIComponent(target.trace_id));if(!current())return;if(trace.id!==target.trace_id||trace.project_id!==project||target.span_id!=null&&!trace.spans.some(span=>span.id===target.span_id))throw new Error('Источник очереди не совпадает');const view=traceView(trace,0,false,target.span_id??null,queue.assignments[index]||null);view.open=true;content.replaceChildren(node('p',target.span_id==null?'Проверяется весь вызов':'Проверяется шаг '+target.span_id),view);}catch(error){if(current())message.textContent=error.message;}finally{source.disabled=false;}};
  });
  const lifecycle=node('button',queue.archived?'Восстановить очередь':'В архив'),lifecycleMessage=node('p');body.append(lifecycle,lifecycleMessage);lifecycle.onclick=async()=>{if(!current()||lifecycle.disabled)return;lifecycle.disabled=true;try{const updated=await api('observability/review-queues/'+encodeURIComponent(queue.id)+'/lifecycle',{base_version:queue.version,archived:!queue.archived});if(!current())return;if(updated.id!==queue.id||updated.project_id!==project||updated.version!==queue.version+1||updated.archived!==!queue.archived||JSON.stringify(updated.targets)!==JSON.stringify(queue.targets)||JSON.stringify(updated.assignments)!==JSON.stringify(queue.assignments)||JSON.stringify(updated.completions)!==JSON.stringify(queue.completions))throw new Error('Архивная очередь не совпадает');showReviewQueue(updated);}catch(error){if(current())lifecycleMessage.textContent=error.message;}finally{lifecycle.disabled=false;}};
  const download=node('button','Скачать очередь CSV'),exportMessage=node('p');body.append(download,exportMessage);download.onclick=()=>{if(!current())return;try{const csv=reviewQueueExportCsv(queue),url=URL.createObjectURL(new Blob(['\ufeff',csv],{type:'text/csv;charset=utf-8'})),link=node('a');link.href=url;link.download='review-queue-'+queue.id+'-v'+queue.version+'.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);exportMessage.textContent='Выгружена показанная версия очереди '+queue.version;}catch(error){exportMessage.textContent=error.message;}};
  body.append(reviewQueueHistoryPanel(queue,current));
}
async function loadReviewQueues(offset=0){
  const project=reviewQueueProject,epoch=++reviewQueueEpoch,current=()=>reviewQueueCurrent(epoch,project),body=$('review-queues-list');reviewQueueViewEpoch++;$('review-queue-detail').replaceChildren();body.replaceChildren(node('p','Загружаю очереди…'));
  const status=$('review-queues-status').value,reviewer=$('review-queues-reviewer').value,query={project_id:project,offset:String(offset),limit:'20'};const archived=$('review-queues-archive').value;if(archived)query.archived=archived;const name=$('review-queues-name').value;if(name)query.name=name;if(status)query.status=status;if(reviewer)query.reviewer=reviewer;
  try{const packet=await api('observability/review-queues?'+new URLSearchParams(query));if(!current())return;if(packet.filters?.status!==(status||null)||packet.filters?.reviewer!==(reviewer||null)||packet.filters?.name!==(name||null)||packet.filters?.archived!==(archived===''?null:archived==='true'))throw new Error('Фильтры очередей не совпадают');
    if(packet.kind!=='review_queue_catalog'||packet.project_id!==project||packet.offset!==offset||packet.limit!==20||packet.order!=='id_desc'||packet.provider_calls!==0||!Number.isInteger(packet.total)||packet.total<0||packet.total>1000||!Array.isArray(packet.queues)||packet.queues.length!==Math.min(20,Math.max(0,packet.total-offset))||packet.has_more!==(offset+packet.queues.length<packet.total)||new Set(packet.queues.map(item=>item.id)).size!==packet.queues.length||packet.queues.some((item,index)=>typeof item.id!=='string'||index>0&&packet.queues[index-1].id<=item.id))throw new Error('Некорректный каталог очередей');
    body.replaceChildren(node('p','Найдено очередей: '+packet.total));
    for(const item of packet.queues){if(item.project_id!==project||typeof item.id!=='string'||!Number.isInteger(item.target_count)||item.target_count<1||item.target_count>200)throw new Error('Некорректная очередь в каталоге');const open=node('button','Открыть '+item.name);body.append(node('p',item.name+' · пунктов: '+item.target_count+' · проверено: '+item.completed_count),open);open.onclick=async()=>{if(!current()||open.disabled)return;open.disabled=true;const selection=++reviewQueueViewEpoch;try{const queue=await api('observability/review-queues/'+encodeURIComponent(item.id));if(!current()||selection!==reviewQueueViewEpoch)return;if(queue.id!==item.id||queue.project_id!==project)throw new Error('Выбранная очередь не совпадает');showReviewQueue(queue);}catch(error){if(current()&&selection===reviewQueueViewEpoch)body.append(node('p',error.message));}finally{open.disabled=false;}};}
    const prev=node('button','Предыдущие очереди'),next=node('button','Следующие очереди');prev.disabled=offset===0;next.disabled=!packet.has_more;prev.onclick=()=>{if(current())return loadReviewQueues(Math.max(0,offset-20));};next.onclick=()=>{if(current())return loadReviewQueues(offset+20);};body.append(prev,next);
  }catch(error){if(current())body.replaceChildren(node('p',error.message));}
}
$('review-queues-open').onclick=()=>{reviewQueueProject=$('project').value;$('review-queues-dialog').showModal();loadReviewQueues();};
$('review-queues-close').onclick=()=>$('review-queues-dialog').close();
$('review-queues-dialog').onclose=()=>{reviewQueueEpoch++;};
$('review-queues-archive').onchange=()=>loadReviewQueues();
$('review-queues-status').onchange=()=>loadReviewQueues();
$('review-queues-name').oninput=()=>{reviewQueueEpoch++;reviewQueueViewEpoch++;};
$('review-queues-reviewer').oninput=()=>{reviewQueueEpoch++;reviewQueueViewEpoch++;};
$('review-queues-refresh').onclick=()=>{reviewQueueProject=$('project').value;return loadReviewQueues();};
function feedbackHistoryPanel(traceId,current,openVersion){
  const panel=node('details'),body=node('div'),load=node('button','Показать версии оценок');
  panel.append(node('summary','Список версий ручных оценок'),load,body);let epoch=0;
  async function page(offset){
    if(!current()||load.disabled)return;load.disabled=true;const token=++epoch;
    const active=()=>current()&&epoch===token;
    try{
      const packet=await api('observability/traces/'+encodeURIComponent(traceId)+'/feedback/versions?offset='+offset+'&limit=20');
      if(!active())return;
      if(packet.kind!=='feedback_history'||packet.trace_id!==traceId||packet.order!=='version_desc'||packet.provider_calls!==0||packet.offset!==offset||packet.limit!==20||!Number.isInteger(packet.total)||packet.total<0||packet.total>2000||packet.latest_version!==packet.total||!Array.isArray(packet.versions)||packet.versions.length!==Math.min(20,Math.max(0,packet.total-offset))||packet.has_more!==(offset+packet.versions.length<packet.total)||packet.versions.some((row,index)=>row.version!==packet.total-offset-index||!Number.isInteger(row.saved_ms)||row.saved_ms<0||!Number.isInteger(row.annotation_count)||row.annotation_count<0||!Number.isInteger(row.active_count)||row.active_count<0||row.active_count>row.annotation_count))throw new Error('Некорректная история оценок');
      body.replaceChildren(node('p','Всего версий: '+packet.total));
      for(const row of packet.versions){
        const line=node('div'),open=node('button','Открыть версию '+row.version);
        line.append(node('p','Версия '+row.version+' · '+new Date(row.saved_ms).toLocaleString()+' · действующих оценок: '+row.active_count+' из '+row.annotation_count),open);
        open.onclick=async()=>{if(!active()||open.disabled)return;open.disabled=true;try{await openVersion(row.version,active);}catch(error){if(active())body.append(node('p',error.message));}finally{open.disabled=false;}};body.append(line);
      }
      const prev=node('button','Предыдущие версии'),next=node('button','Следующие версии');prev.disabled=offset===0;next.disabled=!packet.has_more;
      prev.onclick=()=>{if(active())return page(Math.max(0,offset-20));};next.onclick=()=>{if(active())return page(offset+20);};body.append(prev,next);
    }catch(error){if(active())body.replaceChildren(node('p',error.message));}finally{load.disabled=false;}
  }
  load.onclick=()=>page(0);return panel;
}
function feedbackPanel(trace,initialSpan=null,initialReviewer=null){
  const panel=node('details'),heading=node('summary','Ручные оценки и исправления');panel.className='trace-feedback';panel.append(heading);
  const content=node('div');panel.append(content);let receipt=null,editing=null,latestVersion=0,view=0,operation=0,reviewSearch="",reviewStatus="all";
  const endpoint='observability/traces/'+encodeURIComponent(trace.id)+'/feedback';
  function labeled(text,control){const label=node('label',text);label.append(control);return label;}
  function draw(){
    const selectedView=++view;
    const target=node('select');target.append(new Option('Весь запуск',''));
    for(const span of trace.spans)target.append(new Option(span.name+' · шаг '+span.id,String(span.id)));
    const reviewer=node('input');reviewer.value=editing?.author||initialReviewer||'Я';reviewer.maxLength=200;
    const metric=node('input');metric.value=editing?.metric||'Качество';metric.maxLength=100;
    const type=node('select');type.append(new Option('Число','number'),new Option('Категория','category'),new Option('Без оценки','none'));type.value=editing?.category!=null?'category':editing&&editing.metric==null?'none':'number';
    const score=node('input');score.value=editing?.value??editing?.category??'';
    const comment=node('textarea');comment.value=editing?.comment||'';comment.maxLength=16000;
    const correction=node('textarea');correction.value=editing?.correction||'';correction.maxLength=65536;
    target.value=editing?(editing.span_id==null?'':String(editing.span_id)):(initialSpan==null?'':String(initialSpan));target.disabled=reviewer.disabled=!!editing;
    const metricLabel=labeled('Название оценки',metric),scoreLabel=labeled('Значение',score);
    function changeType(){metricLabel.hidden=scoreLabel.hidden=type.value==='none';score.type=type.value==='number'?'number':'text';score.step='any';}
    type.onchange=changeType;changeType();
    const form=node('form');form.className='feedback-form';
    form.append(labeled('Что оцениваем',target),labeled('Проверяющий',reviewer),labeled('Тип оценки',type),metricLabel,scoreLabel,labeled('Комментарий',comment),labeled('Исправленный ответ',correction));
    const save=node('button',editing?'Сохранить правку':'Добавить оценку');save.type='submit';form.append(save);
    const cancel=node('button','Отменить правку');cancel.type='button';cancel.hidden=!editing;cancel.onclick=()=>{editing=null;draw();};form.append(cancel);
    const message=node('p');message.setAttribute('role','status');
    form.onsubmit=async event=>{
      event.preventDefault();if(!panel.open||selectedView!==view||save.disabled)return;save.disabled=true;const token=++operation;
      try{
        if(type.value==='number'&&(score.value.trim()===''||!Number.isFinite(Number(score.value))))throw new Error('Введите число для оценки.');
        const saved=await api(endpoint,{base_version:receipt.version,annotation:{id:editing?.id||null,span_id:target.value===''?null:Number(target.value),author:reviewer.value,metric:type.value==='none'?null:metric.value,value:type.value==='number'?Number(score.value):null,category:type.value==='category'?score.value:null,comment:comment.value||null,correction:correction.value||null,deleted:false}});
        if(!panel.open||selectedView!==view||token!==operation)return;
        if(saved.trace_id!==trace.id||!Number.isInteger(saved.version)||saved.version<1)throw new Error('Некорректная версия оценок');
        receipt=saved;latestVersion=receipt.version;editing=null;draw();notify('Оценка сохранена');
      }catch(error){if(panel.open&&selectedView===view&&token===operation)message.textContent=error.message;}finally{save.disabled=false;}
    };
    const refresh=node('button','Обновить оценки');refresh.onclick=load;
    const versions=feedbackHistoryPanel(trace.id,()=>panel.open&&selectedView===view,async(version,current)=>{
      const token=++operation;const selected=await api(endpoint+'?version='+version);if(!current()||token!==operation)return;
      if(selected.trace_id!==trace.id||selected.version!==version)throw new Error('Версия оценок не совпадает');
      receipt=selected;editing=null;draw();
    });

    const history=node('select');for(let version=latestVersion;version>=0;version--)history.append(new Option('Версия '+version+(version===latestVersion?' · текущая':''),String(version)));history.value=String(receipt.version);
    history.onchange=async()=>{if(!panel.open||selectedView!==view)return;const version=Number(history.value),token=++operation;try{const selected=await api(endpoint+'?version='+version);if(!panel.open||selectedView!==view||token!==operation)return;if(selected.trace_id!==trace.id||selected.version!==version)throw new Error('Версия оценок не совпадает');receipt=selected;editing=null;draw();}catch(error){if(panel.open&&selectedView===view&&token===operation)message.textContent=error.message;}};
    const historical=receipt.version!==latestVersion;form.hidden=historical;

    const exportButton=node('button','Экспорт оценок');exportButton.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(receipt,null,2)],{type:'application/json'}));const link=node('a');link.href=url;link.download='feedback-'+trace.id+'-v'+receipt.version+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
    const list=node('div');list.className='feedback-list';
    const reviewFilter=node('input');reviewFilter.type='search';reviewFilter.value=reviewSearch;reviewFilter.setAttribute('aria-label','Поиск ручных оценок');
    const statusFilter=node('select');statusFilter.setAttribute('aria-label','Состояние ручных оценок');statusFilter.append(new Option('Все оценки','all'),new Option('Действующие','active'),new Option('Удалённые','deleted'));statusFilter.value=reviewStatus;
    const reviewCount=node('p'),reviewRows=[];
    function filterReviews(){const query=reviewSearch.trim().toLocaleLowerCase();let count=0;for(const {row,annotation} of reviewRows){const matches=(reviewStatus==='all'||(reviewStatus==='deleted')===!!annotation.deleted)&&(!query||[annotation.author,annotation.metric,annotation.category,annotation.comment,annotation.correction].some(value=>typeof value==='string'&&value.toLocaleLowerCase().includes(query)));row.hidden=!matches;if(matches)count++;}reviewCount.textContent='Показано оценок: '+count+' из '+reviewRows.length;}
    reviewFilter.oninput=()=>{reviewSearch=reviewFilter.value;filterReviews();};statusFilter.onchange=()=>{reviewStatus=statusFilter.value;filterReviews();};

    list.append(reviewFilter,statusFilter,reviewCount);
    if(receipt.summaries.length)list.append(node('p','Сводка всей версии (без фильтров)'));
    for(const summary of receipt.summaries){list.append(node('p',(summary.span_id==null?'Весь запуск':'Шаг '+summary.span_id)+' · '+summary.metric+' · '+(summary.mean==null?'':('среднее '+summary.mean+' ('+summary.count+') '))+Object.entries(summary.categories).map(([label,count])=>label+': '+count).join(', ')));}
    for(const annotation of receipt.annotations){
      const row=node('details');row.className='feedback-entry';row.append(node('summary',annotation.author+' · '+(annotation.span_id==null?'Весь запуск':'Шаг '+annotation.span_id)+(annotation.deleted?' · удалена':'')));
      if(annotation.metric)row.append(node('p',annotation.metric+': '+(annotation.value??annotation.category)));
      if(annotation.comment)row.append(node('pre',annotation.comment));
      if(annotation.correction)row.append(node('p','Исправленный ответ'),node('pre',annotation.correction));
      const edit=node('button','Изменить');edit.hidden=annotation.deleted||historical;edit.onclick=()=>{editing=annotation;draw();};
      const remove=node('button',annotation.deleted?'Восстановить':'Убрать оценку');remove.hidden=historical;remove.onclick=async()=>{if(!panel.open||selectedView!==view||remove.disabled)return;remove.disabled=true;const token=++operation;try{const saved=await api(endpoint,{base_version:receipt.version,annotation:{...annotation,deleted:!annotation.deleted}});if(!panel.open||selectedView!==view||token!==operation)return;if(saved.trace_id!==trace.id||!Number.isInteger(saved.version)||saved.version<1)throw new Error('Некорректная версия оценок');receipt=saved;latestVersion=receipt.version;editing=null;draw();}catch(error){if(panel.open&&selectedView===view&&token===operation)message.textContent=error.message;}finally{remove.disabled=false;}};
      row.append(edit,remove);list.append(row);reviewRows.push({row,annotation});
    }
    filterReviews();
    content.replaceChildren(node('p','Версия '+receipt.version+'. Имя проверяющего указывается вручную. Введённые комментарии и исправления сохраняются на этом компьютере.'),refresh,labeled('История правок',history),versions,exportButton,historical?node('p','Прошлая версия открыта для просмотра. Для правок выберите текущую.'):node('span'),list,form,message);
  }
  async function load(){const token=++operation;try{const selected=await api(endpoint);if(!panel.open||token!==operation)return;if(selected.trace_id!==trace.id||!Number.isInteger(selected.version)||selected.version<0)throw new Error('Некорректная версия оценок');receipt=selected;latestVersion=receipt.version;editing=null;draw();}catch(error){if(!panel.open||token!==operation)return;content.replaceChildren(node('p',error.message));const retry=node('button','Повторить');retry.onclick=load;content.append(retry);}}
  panel.ontoggle=()=>{if(!panel.open){operation++;view++;}else if(!receipt){load();}else{draw();}};return panel;
}
function guardrailReceiptRows(receipt){
  if(!receipt)return [];
  if(receipt.kind!=='local_guardrail'||receipt.schema_version!==1||receipt.provider_calls!==0||receipt.content_captured!==false||!['input','output'].includes(receipt.stage)||!['block','observe'].includes(receipt.action)||!/^[a-f0-9]{64}$/.test(receipt.policy_sha256)||typeof receipt.passed!=='boolean'||receipt.blocked!==(!receipt.passed&&receipt.action==='block')||!Array.isArray(receipt.rules)||receipt.rules.length<1||receipt.rules.length>32||receipt.rules.some(rule=>!/^[A-Za-z0-9._-]{1,80}$/.test(rule.rule_id)||!['min_bytes','max_bytes','json_valid','forbidden_substrings','required_substrings'].includes(rule.kind)||typeof rule.passed!=='boolean')||new Set(receipt.rules.map(rule=>rule.rule_id)).size!==receipt.rules.length||receipt.rules.every(rule=>rule.passed)!==receipt.passed)return [node('p','Некорректные результаты правил.')];
  const detail=node('details');detail.append(node('summary','Правила: '+receipt.rules.length+' · '+(receipt.passed?'успешно':receipt.blocked?'заблокировано':'нарушение')),node('p',(receipt.stage==='input'?'Вход':'Ответ')+' · '+(receipt.action==='block'?'Блокировка':'Наблюдение')+' · '+receipt.policy_sha256));for(const rule of receipt.rules)detail.append(node('p',rule.rule_id+' · '+rule.kind+': '+(rule.passed?'успешно':'нарушение')));return [detail];
}
let qualityCaptureEpoch=0,qualityCaptureContext=null,qualityCaptureSaving=false;
async function openQualityCapture(trace){
 const epoch=++qualityCaptureEpoch,project=$('project').value,session=active;qualityCaptureContext=null;
 const dialog=$('quality-capture-dialog'),message=$('quality-capture-message');dialog.showModal();$('quality-capture-save').disabled=true;
 for(const id of ['quality-capture-input','quality-capture-output','quality-capture-reference'])$(id).value='';
 message.replaceChildren(node('p','Проверяю завершённую трассу…'));
 const current=()=>epoch===qualityCaptureEpoch&&dialog.open&&$('project').value===project&&active===session;
 try{if(trace.project_id!==project||trace.status!=='completed')throw new Error('Выберите завершённую трассу текущего проекта.');const selection=await api('observability/online-selections',{project_id:project,trace_id:trace.id});if(!current())return;
  if(selection.kind!=='online_evaluation_selection'||selection.schema_version!==2||selection.project_id!==project||selection.trace_id!==trace.id||!/^[0-9a-f]{64}$/.test(selection.trace_sha256)||!/^[0-9a-f]{64}$/.test(selection.selection_sha256)||selection.provider_calls!==0||selection.automatic_execution!==false)throw new Error('Некорректная версия трассы');
  qualityCaptureContext={epoch,project,session,trace_id:trace.id,trace_sha256:selection.trace_sha256};message.replaceChildren(node('p','Трасса проверена. Введите текст для оценки.'));$('quality-capture-save').disabled=false;
 }catch(error){if(current())message.replaceChildren(node('p',error.message));}
}
$('quality-capture-close').onclick=()=>$('quality-capture-dialog').close();
$('quality-capture-dialog').onclose=()=>{qualityCaptureEpoch++;qualityCaptureContext=null;for(const id of ['quality-capture-input','quality-capture-output','quality-capture-reference'])$(id).value='';$('quality-capture-message').replaceChildren();};
$('quality-capture-save').onclick=async()=>{
 const context=qualityCaptureContext;if(!context||qualityCaptureSaving)return;
 const input=$('quality-capture-input').value,output=$('quality-capture-output').value,reference=$('quality-capture-reference').value||null;
 const current=()=>qualityCaptureContext===context&&$('quality-capture-dialog').open&&$('project').value===context.project&&active===context.session&&$('quality-capture-input').value===input&&$('quality-capture-output').value===output&&($('quality-capture-reference').value||null)===reference;
 if(!current())return;const message=$('quality-capture-message'),bytes=value=>new TextEncoder().encode(value).length;
 if(bytes(input)>16000||!output.trim()||bytes(output)>64000||(reference!==null&&bytes(reference)>64000)){message.replaceChildren(node('p','Укажите непустой ответ. Лимиты: вопрос 16000 байт, ответ и эталон по 64000 байт.'));return;}
 qualityCaptureSaving=true;$('quality-capture-save').disabled=true;
 try{const source={project_id:context.project,trace_id:context.trace_id,trace_sha256:context.trace_sha256,input,output,reference};const saved=await api('observability/online-quality-sources',source);if(!current())return;
  if(saved.kind!=='online_quality_source'||saved.schema_version!==1||!/^[0-9a-f]{64}$/.test(saved.source_sha256)||saved.answer_source!=='caller_supplied'||saved.trace_content_verified!==false||!saved.source||Object.keys(source).some(key=>saved.source[key]!==source[key]))throw new Error('Некорректное подтверждение сохранения текста');
  const view=node('button','Открыть модельные оценки');view.onclick=()=>{if(current()){$('quality-capture-dialog').close();$('quality-jobs-open').onclick();}};
  message.replaceChildren(node('p','Текст сохранён. Выбранные модельные правила могут создать задания в очереди.'),view);
 }catch(error){if(current())message.replaceChildren(node('p',error.message));}finally{qualityCaptureSaving=false;if(qualityCaptureContext===context)$('quality-capture-save').disabled=false;}
};
let qualityJobsEpoch=0;
async function loadQualityJobs(offset=0){
 const epoch=++qualityJobsEpoch,project=$('project').value,body=$('quality-jobs-body');const current=()=>epoch===qualityJobsEpoch&&$('quality-jobs-dialog').open&&$('project').value===project;
 body.replaceChildren(node('p','Загружаю модельные оценки…'));
 try{const packet=await api('observability/online-quality-jobs?'+new URLSearchParams({project_id:project,offset,limit:20}));if(!current())return;
  const sha=value=>typeof value==='string'&&/^[0-9a-f]{64}$/.test(value),states=['pending','running','completed','failed','interrupted'];
  if(packet.kind!=='online_quality_job_catalog'||packet.project_id!==project||packet.offset!==offset||packet.limit!==20||!Number.isSafeInteger(packet.total)||packet.total<0||packet.total>1000||packet.order!=='id_ascending'||packet.provider_calls!==0||packet.automatic_execution!==false||!Array.isArray(packet.jobs)||packet.jobs.length!==Math.min(20,Math.max(0,packet.total-offset))||packet.has_more!==(offset+packet.jobs.length<packet.total))throw new Error('Некорректная страница модельных оценок');
  let previous='';for(const row of packet.jobs){if(!sha(row.id)||row.id<=previous||row.project_id!==project||!sha(row.source_sha256)||typeof row.provider!=='string'||typeof row.model!=='string'||!states.includes(row.status))throw new Error('Некорректное модельное задание');previous=row.id;const result=row.result;
   if(['pending','running'].includes(row.status)){if(result!==null)throw new Error('Неожиданный результат задания');}
   else if(!result||result.kind!=='online_quality_job_result'||result.schema_version!==1||result.id!==row.id||result.status!==row.status||!sha(result.result_sha256)||result.automatic_execution!==true||(row.status==='completed'&&(!(typeof result.judge_id==='string'&&/^[a-zA-Z0-9-]{1,80}$/.test(result.judge_id))||!sha(result.judge_receipt_sha256)))||(row.status!=='completed'&&(result.judge_id!==null||result.judge_receipt_sha256!==null)))throw new Error('Некорректный результат модельного задания');
  }
  body.replaceChildren(node('p','Всего заданий: '+packet.total));if(!packet.jobs.length)body.append(node('p','Модельных заданий пока нет.'));
  for(const row of packet.jobs){const item=node('div'),label={pending:'Ожидает',running:'Выполняется',completed:'Завершено',failed:'Ошибка',interrupted:'Прервано'}[row.status];item.append(node('p',label+' · '+row.provider+' / '+row.model));
   if(row.status==='interrupted')item.append(node('p','Запрос мог дойти до провайдера. Автоматического повторения нет.'));
   if(row.status==='completed'){const open=node('button','Показать балл и объяснение'),detail=node('div');item.append(open,detail);open.onclick=async()=>{if(!current()||open.disabled)return;open.disabled=true;try{const verdict=await api('evaluation/judge/'+row.result.judge_id);if(!current())return;if(verdict.kind!=='llm_judge'||verdict.id!==row.result.judge_id||verdict.project_id!==project||verdict.quality_source_sha256!==row.source_sha256||verdict.receipt_sha256!==row.result.judge_receipt_sha256||typeof verdict.score!=='number'||!Number.isFinite(verdict.score)||verdict.score<0||verdict.score>1||typeof verdict.reason!=='string'||!verdict.reason.trim()||verdict.reason.length>8000||verdict.automatic_promotion!==false||verdict.answer_source!=='caller_supplied'||verdict.trace_content_verified!==false)throw new Error('Некорректный вердикт модели');detail.replaceChildren(node('p','Балл: '+verdict.score+' из 1'),node('p',verdict.reason));}catch(error){if(current())detail.replaceChildren(node('p',error.message));}finally{if(current())open.disabled=false;}};}
   body.append(item);
  }
  const prev=node('button','Предыдущая страница'),next=node('button','Следующая страница');prev.disabled=offset===0;next.disabled=!packet.has_more;prev.onclick=()=>{if(current()&&!prev.disabled)return loadQualityJobs(Math.max(0,offset-20));};next.onclick=()=>{if(current()&&!next.disabled)return loadQualityJobs(offset+20);};body.append(prev,next);
 }catch(error){if(current())body.replaceChildren(node('p',error.message));}
}
$('quality-jobs-open').onclick=()=>{$('quality-jobs-dialog').showModal();return loadQualityJobs();};
$('quality-jobs-refresh').onclick=()=>loadQualityJobs();
$('quality-jobs-close').onclick=()=>$('quality-jobs-dialog').close();
$('quality-jobs-dialog').onclose=()=>{qualityJobsEpoch++;$('quality-jobs-body').replaceChildren();};
let onlineRulesEpoch=0,onlineRuleSaving=false;
function validOnlineBinding(packet,project,id){
  const binding=packet.binding,sha=value=>typeof value==='string'&&/^[0-9a-f]{64}$/.test(value);
  if(packet.kind!=='online_evaluation_binding_status'||packet.project_id!==project||packet.rule_id!==id||packet.provider_calls!==0||packet.automatic_execution!==false||!(binding===null||(binding.kind==='online_evaluation_binding'&&binding.project_id===project&&binding.rule_id===id&&sha(binding.rule_sha256)&&sha(binding.binding_sha256)&&Number.isSafeInteger(binding.version)&&binding.version>0&&binding.version<=1000&&typeof binding.active==='boolean')))throw new Error('Некорректное состояние правила');
  return binding;
}
async function loadOnlineRules(offset=0){
  const epoch=++onlineRulesEpoch,project=$('project').value,body=$('online-rules-body');
  const current=()=>epoch===onlineRulesEpoch&&$('online-rules-dialog').open&&$('project').value===project;
  body.replaceChildren(node('p','Загружаю правила…'));
  try{
    const packet=await api('observability/online-rules?'+new URLSearchParams({project_id:project,offset,limit:20}));if(!current())return;
    if(packet.kind!=='online_evaluation_rule_catalog'||packet.project_id!==project||packet.offset!==offset||packet.limit!==20||!Number.isSafeInteger(packet.total)||packet.total<0||packet.total>1000||packet.order!=='rule_sha256_ascending'||packet.provider_calls!==0||packet.automatic_execution!==false||!Array.isArray(packet.rules)||packet.rules.length!==Math.min(20,Math.max(0,packet.total-offset))||packet.has_more!==(offset+packet.rules.length<packet.total))throw new Error('Некорректная страница правил');
    let previous='';for(const row of packet.rules){const rule=row.rule;if(!/^[0-9a-f]{64}$/.test(row.rule_sha256)||row.rule_sha256<=previous||!rule||rule.project_id!==project||typeof rule.id!=='string'||!/^[a-zA-Z0-9._-]{1,100}$/.test(rule.id)||typeof rule.evaluator_id!=='string'||!Number.isSafeInteger(rule.evaluator_version)||rule.evaluator_version<1||typeof rule.enabled!=='boolean'||!Number.isFinite(rule.sample_rate)||rule.sample_rate<0||rule.sample_rate>1)throw new Error('Некорректная версия правила');previous=row.rule_sha256;}
    const bindings=new Map();await Promise.all([...new Set(packet.rules.map(row=>row.rule.id))].map(async id=>{const status=await api('observability/online-rule-bindings?'+new URLSearchParams({project_id:project,rule_id:id}));bindings.set(id,validOnlineBinding(status,project,id));}));if(!current())return;
    body.replaceChildren(node('p','Сохранённых версий: '+packet.total));if(!packet.rules.length)body.append(node('p','Правил пока нет.'));
    for(const row of packet.rules){const rule=row.rule,binding=bindings.get(rule.id),active=binding?.active&&binding.rule_sha256===row.rule_sha256;
      const item=node('div');item.append(node('p',rule.id+' · '+Math.round(rule.sample_rate*10000)/100+'% трасс · '+(rule.evaluator_id.startsWith('model_quality.')?'Модельная оценка':rule.evaluator_id)+' v'+rule.evaluator_version+' · '+(active?'Включено':binding?.active?'Включена другая версия':'Выключено')));
      const toggle=node('button',active?'Отключить':'Включить эту версию');toggle.disabled=!rule.enabled||binding?.version===1000;
      toggle.onclick=async()=>{if(!current()||toggle.disabled)return;toggle.disabled=true;try{const changed=await api('observability/online-rule-bindings',{rule_sha256:row.rule_sha256,base_version:binding?.version??0,active:!active});if(!current())return;if(changed.kind!=='online_evaluation_binding'||changed.project_id!==project||changed.rule_id!==rule.id||changed.rule_sha256!==row.rule_sha256||changed.version!==(binding?.version??0)+1||changed.active!==!active)throw new Error('Некорректное подтверждение изменения');return loadOnlineRules(offset);}catch(error){if(current()){item.append(node('p',error.message+' · Обновите правила перед повторной попыткой.'));}}};item.append(toggle);body.append(item);
    }
    const prev=node('button','Предыдущая страница'),next=node('button','Следующая страница');prev.disabled=offset===0;next.disabled=!packet.has_more;prev.onclick=()=>{if(current()&&!prev.disabled)return loadOnlineRules(Math.max(0,offset-20));};next.onclick=()=>{if(current()&&!next.disabled)return loadOnlineRules(offset+20);};body.append(prev,next);
  }catch(error){if(current())body.replaceChildren(node('p',error.message));}
}
function updateOnlineRuleKind(){const model=$('online-rule-kind').value==='model_quality';$('online-rule-model-fields').hidden=!model;const selected=settings();$('online-rule-model-label').textContent='Оценщик: '+selected.provider+' / '+(selected.model||'Модель не выбрана')+' · Chat без записи файлов';}
$('online-rule-kind').onchange=updateOnlineRuleKind;
$('online-rules-open').onclick=()=>{updateOnlineRuleKind();$('online-rules-dialog').showModal();return loadOnlineRules();};
$('online-rules-close').onclick=()=>$('online-rules-dialog').close();
$('online-rules-dialog').onclose=()=>{onlineRulesEpoch++;$('online-rules-body').replaceChildren();$('online-rule-message').replaceChildren();};
$('online-rules-refresh').onclick=()=>loadOnlineRules();
function onlineRuleSettings(){return {...settings(),mode:'chat',allow_writes:false,max_output_tokens:Math.min(settings().max_output_tokens,4096)};}
function canonicalOnlineValue(value){if(Array.isArray(value))return value.map(canonicalOnlineValue);if(value&&typeof value==='object')return Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonicalOnlineValue(value[key])]));return value;}
$('online-rule-save').onclick=async()=>{
 if(onlineRuleSaving||!$('online-rules-dialog').open)return;
 const project=$('project').value,id=$('online-rule-id').value.trim(),input=$('online-rule-rate').value,rate=Number(input),epoch=onlineRulesEpoch,kind=$('online-rule-kind').value,rubric=$('online-rule-rubric').value.trim(),selected=kind==='model_quality'?onlineRuleSettings():null;
 const encoded=value=>JSON.stringify(canonicalOnlineValue(value));
 const current=()=>$('online-rules-dialog').open&&$('project').value===project&&onlineRulesEpoch===epoch&&$('online-rule-id').value.trim()===id&&$('online-rule-rate').value===input&&$('online-rule-kind').value===kind&&$('online-rule-rubric').value.trim()===rubric&&(selected===null||encoded(onlineRuleSettings())===encoded(selected));
 const message=$('online-rule-message');if(!/^[a-zA-Z0-9._-]{1,100}$/.test(id)||!input.trim()||!Number.isFinite(rate)||rate<0||rate>100){message.replaceChildren(node('p','Укажите имя латиницей и долю от 0 до 100%.'));return;}
 if(!['trace_health','model_quality'].includes(kind)||(kind==='model_quality'&&(!rubric||new TextEncoder().encode(rubric).length>16000||!selected.model))){message.replaceChildren(node('p','Выберите модель и задайте критерий до 16000 байт.'));return;}
 onlineRuleSaving=true;$('online-rule-save').disabled=true;
 try{let evaluator='trace_health';
  if(kind==='model_quality'){const config=await api('observability/online-model-evaluators',{settings:selected,rubric});if(!current())return;
   if(config.kind!=='online_model_evaluator'||config.schema_version!==1||!/^[0-9a-f]{64}$/.test(config.evaluator_sha256)||config.evaluator_id!=='model_quality.'+config.evaluator_sha256||config.evaluator_version!==1||config.definition?.rubric!==rubric||!config.definition?.settings||Object.keys(selected).some(key=>encoded(config.definition.settings[key])!==encoded(selected[key])))throw new Error('Некорректное подтверждение настройки оценщика');evaluator=config.evaluator_id;
  }
  const rule={id,project_id:project,evaluator_id:evaluator,evaluator_version:1,sample_rate:rate/100,enabled:true};const saved=await api('observability/online-rules',rule);if(!current())return;
  if(saved.kind!=='online_evaluation_rule'||saved.schema_version!==1||!/^[0-9a-f]{64}$/.test(saved.rule_sha256)||!saved.rule||Object.keys(rule).some(key=>saved.rule[key]!==rule[key]))throw new Error('Некорректное подтверждение сохранения');
  message.replaceChildren(node('p','Версия сохранена. Включите её в списке правил.'));return loadOnlineRules();
 }catch(error){if(current())message.replaceChildren(node('p',error.message));}finally{onlineRuleSaving=false;$('online-rule-save').disabled=false;}
};
let onlineJobsEpoch=0;
async function loadOnlineJobs(offset=0){
  const epoch=++onlineJobsEpoch,project=$('project').value,dialog=$('online-jobs-dialog'),body=$('online-jobs-body');
  const current=()=>epoch===onlineJobsEpoch&&dialog.open&&$('project').value===project;
  body.replaceChildren(node('p','Загружаю очередь…'));
  try{
    const packet=await api('observability/online-jobs?'+new URLSearchParams({project_id:project,offset,limit:20}));
    if(!current())return;
    const sha=value=>typeof value==='string'&&/^[0-9a-f]{64}$/.test(value),integer=value=>Number.isSafeInteger(value)&&value>=0;
    if(packet.kind!=='online_evaluation_job_catalog'||packet.project_id!==project||packet.offset!==offset||packet.limit!==20||!integer(packet.total)||packet.total>1000||packet.order!=='selection_sha256_ascending'||packet.provider_calls!==0||packet.automatic_execution!==false||!Array.isArray(packet.jobs)||packet.jobs.length>20||packet.jobs.length!==Math.min(20,Math.max(0,packet.total-offset))||packet.has_more!==(offset+packet.jobs.length<packet.total))throw new Error('Некорректная страница очереди');
    let previous='';
    for(const row of packet.jobs){const job=row.job,result=row.result;
      if(!job||job.kind!=='online_evaluation_job'||job.schema_version!==1||job.project_id!==project||typeof job.trace_id!=='string'||!/^trace-[a-zA-Z0-9-]+$/.test(job.trace_id)||!sha(job.trace_sha256)||!sha(job.selection_sha256)||job.selection_sha256<=previous||job.status!=='pending'||job.provider_calls!==0||!['pending','completed','failed'].includes(row.status))throw new Error('Некорректное задание очереди');
      previous=job.selection_sha256;
      if(row.status==='pending'){if(result!==null)throw new Error('Ожидающее задание содержит результат');}
      else if(!result||result.kind!=='online_evaluation_job_result'||result.schema_version!==1||result.selection_sha256!==job.selection_sha256||result.status!==row.status||!sha(result.job_sha256)||!sha(result.result_sha256)||result.provider_calls!==0||typeof result.automatic_execution!=='boolean'||(row.status==='completed'&&(!sha(result.assessment_sha256)||result.error_code!==null))||(row.status==='failed'&&(!['evaluator_failed','assessment_rejected'].includes(result.error_code)||!(result.assessment_sha256===null||sha(result.assessment_sha256)))))throw new Error('Некорректный результат задания');
    }
    body.replaceChildren(node('p','Всего заданий: '+packet.total));
    if(!packet.jobs.length)body.append(node('p','На этой странице заданий нет.'));
    for(const row of packet.jobs){const item=node('details'),status={pending:'Ожидает',completed:'Завершено',failed:'Ошибка'}[row.status];
      item.append(node('summary',status+' · '+row.job.trace_id));
      if(row.result)item.append(node('p',(row.result.automatic_execution?'Автоматическая обработка':'Ручная обработка')+(row.result.error_code?' · '+({evaluator_failed:'Оценщик завершился с ошибкой',assessment_rejected:'Исходные данные оценки не прошли проверку'}[row.result.error_code]):'')));
      const evidence=node('details');evidence.append(node('summary','Сохранённые данные проверки'),node('pre',JSON.stringify({job:row.job,result:row.result},null,2)));item.append(evidence);body.append(item);
    }
    const prev=node('button','Предыдущая страница'),next=node('button','Следующая страница');prev.disabled=offset===0;next.disabled=!packet.has_more;
    prev.onclick=()=>{if(current()&&!prev.disabled)return loadOnlineJobs(Math.max(0,offset-20));};next.onclick=()=>{if(current()&&!next.disabled)return loadOnlineJobs(offset+20);};body.append(prev,next);
  }catch(error){if(current())body.replaceChildren(node('p',error.message));}
}
$('online-jobs-open').onclick=()=>{$('online-jobs-dialog').showModal();return loadOnlineJobs();};
$('online-jobs-refresh').onclick=()=>loadOnlineJobs();
$('online-jobs-close').onclick=()=>$('online-jobs-dialog').close();
$('online-jobs-dialog').onclose=()=>{onlineJobsEpoch++;$('online-jobs-body').replaceChildren();};
let callbackSummaryEpoch=0;
async function loadCallbackSummary(){
  const epoch=++callbackSummaryEpoch,project=$('project').value,body=$('callback-summary-body'),since=$('callback-summary-since').value,until=$('callback-summary-until').value;
  const current=()=>epoch===callbackSummaryEpoch&&project===$('project').value&&$('callback-summary-dialog').open;body.replaceChildren(node('p','Загружаю сводку…'));
  try{const from=since===''?null:new Date(since).getTime(),to=until===''?null:new Date(until).getTime();if(from!=null&&(!Number.isSafeInteger(from)||from<0)||to!=null&&(!Number.isSafeInteger(to)||to<0)||from!=null&&to!=null&&from>to)throw new Error('Проверьте диапазон дат');const query={project_id:project};if(from!=null)query.since_ms=String(from);if(to!=null)query.until_ms=String(to);const packet=await api('observability/evaluations/summary?'+new URLSearchParams(query));if(!current())return;
    if(packet.kind!=='callback_evaluation_summary'||packet.project_id!==project||packet.since_ms!==from||packet.until_ms!==to||packet.assessment_source!=='caller_reported'||packet.provider_calls!==0||packet.automatic_promotion!==false||['trace_count','selected_tasks','skipped_tasks','completed_assessments','failed_assessments'].some(key=>!Number.isInteger(packet[key])||packet[key]<0||packet[key]>20000000)||!Array.isArray(packet.metrics)||packet.metrics.length>4000)throw new Error('Некорректная сводка оценок');
    const seen=new Set();for(const row of packet.metrics){const key=JSON.stringify([row.evaluator_id,row.evaluator_version,row.metric]);if(seen.has(key)||typeof row.metric!=='string'||!/^[A-Za-z0-9._-]{1,100}$/.test(row.metric)||!Number.isInteger(row.count)||row.count<1||row.count>20000000||['mean','min','max'].some(name=>typeof row[name]!=='number'||!Number.isFinite(row[name])||row[name]<0||row[name]>1)||row.min>row.mean||row.mean>row.max||!(row.evaluator_id===null&&row.evaluator_version===null)&&(!(typeof row.evaluator_id==='string'&&/^[A-Za-z0-9._-]{1,100}$/.test(row.evaluator_id))||!Number.isSafeInteger(row.evaluator_version)||row.evaluator_version<1))throw new Error('Некорректная группа оценок');seen.add(key);}
    if(packet.evaluators!==undefined){if(!Array.isArray(packet.evaluators)||packet.evaluators.length>4000)throw new Error('Некорректные счётчики оценщиков');const identities=new Set();let completed=0,failed=0;for(const row of packet.evaluators){if(!row||typeof row!=='object'||Array.isArray(row)||Object.keys(row).length!==4||!['evaluator_id','evaluator_version','completed_assessments','failed_assessments'].every(key=>Object.prototype.hasOwnProperty.call(row,key))||!(row.evaluator_id===null&&row.evaluator_version===null)&&(!(typeof row.evaluator_id==='string'&&/^[A-Za-z0-9._-]{1,100}$/.test(row.evaluator_id))||!Number.isSafeInteger(row.evaluator_version)||row.evaluator_version<1)||['completed_assessments','failed_assessments'].some(key=>!Number.isInteger(row[key])||row[key]<0||row[key]>20000000)||row.completed_assessments+row.failed_assessments===0)throw new Error('Некорректные счётчики оценщика');const key=JSON.stringify([row.evaluator_id,row.evaluator_version]);if(identities.has(key))throw new Error('Повторная версия оценщика');identities.add(key);completed+=row.completed_assessments;failed+=row.failed_assessments;}if(completed!==packet.completed_assessments||failed!==packet.failed_assessments||packet.metrics.some(row=>!identities.has(JSON.stringify([row.evaluator_id,row.evaluator_version]))))throw new Error('Счётчики оценщиков не совпадают со сводкой');}
    body.replaceChildren(node('p','Вызовов: '+packet.trace_count+' · выбраны для оценки: '+packet.selected_tasks+' · пропущены по отбору: '+packet.skipped_tasks),node('p','Завершённых попыток оценки: '+packet.completed_assessments+' · ошибок оценщиков: '+packet.failed_assessments),node('p','Отбор учитывает успешные вызовы с заданным оценщиком. Пропуск не означает успешную оценку.'));
    let offset=0;const groups=node('div');body.append(groups);function draw(){groups.replaceChildren(node('p','Групп метрик: '+packet.metrics.length));const table=node('table'),header=node('tr');for(const label of ['Оценщик','Версия','Метрика','Оценок','Среднее','Минимум','Максимум'])header.append(node('th',label));table.append(header);for(const row of packet.metrics.slice(offset,offset+50)){const line=node('tr');for(const value of [row.evaluator_id??'Без идентификатора',row.evaluator_version??'—',row.metric,row.count,row.mean,row.min,row.max])line.append(node('td',String(value)));table.append(line);}groups.append(table);const prev=node('button','Предыдущие метрики'),next=node('button','Следующие метрики');prev.disabled=offset===0;next.disabled=offset+50>=packet.metrics.length;prev.onclick=()=>{if(current()){offset=Math.max(0,offset-50);draw();}};next.onclick=()=>{if(current()){offset+=50;draw();}};groups.append(prev,next);}draw();
    const csvButton=node('button','Скачать все группы CSV'),jsonButton=node('button','Скачать сводку JSON');body.append(csvButton,jsonButton);
    function download(format){if(!current())return;let contents;if(format==='json'){contents=JSON.stringify(packet,null,2);}else{const columns=['project_id','since_ms','until_ms','trace_count','selected_tasks','skipped_tasks','completed_assessments','failed_assessments','assessment_source','provider_calls','automatic_promotion','evaluator_id','evaluator_version','metric','count','mean','min','max'];const cell=value=>{let text=value==null?'':String(value);if(typeof value==='string'&&/^[\s]*[=+@-]/.test(text))text="'"+text;return '"'+text.replace(/"/g,'""')+'"';};const rows=packet.metrics.length?packet.metrics:[{}];contents='\ufeff'+[columns.map(cell).join(','),...rows.map(row=>columns.map(key=>cell(Object.prototype.hasOwnProperty.call(row,key)?row[key]:packet[key])).join(','))].join('\r\n')+'\r\n';}const url=URL.createObjectURL(new Blob([contents],{type:format==='csv'?'text/csv;charset=utf-8':'application/json'})),link=node('a');link.href=url;link.download='callback-summary-'+project+'.'+format;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    csvButton.onclick=()=>download('csv');jsonButton.onclick=()=>download('json');
    if(packet.evaluators!==undefined){const attempts=node('details');attempts.append(node('summary','Попытки по версиям оценщиков · '+packet.evaluators.length));const content=node('div');attempts.append(content);body.append(attempts);let page=0;function drawAttempts(){content.replaceChildren(node('p','Завершённые попытки не обязательно имеют баллы. Ошибки показаны независимо от метрик.'));for(const row of packet.evaluators.slice(page,page+50))content.append(node('p',(row.evaluator_id??'Без идентификатора')+' · версия '+(row.evaluator_version??'—')+' · завершено: '+row.completed_assessments+' · ошибок: '+row.failed_assessments));const prev=node('button','Предыдущие оценщики'),next=node('button','Следующие оценщики');prev.disabled=page===0;next.disabled=page+50>=packet.evaluators.length;prev.onclick=()=>{if(current()){page=Math.max(0,page-50);drawAttempts();}};next.onclick=()=>{if(current()){page+=50;drawAttempts();}};content.append(prev,next);}drawAttempts();const exportAttempts=node('button','Скачать попытки оценщиков CSV');attempts.append(exportAttempts);exportAttempts.onclick=()=>{if(!current())return;const columns=['project_id','since_ms','until_ms','assessment_source','provider_calls','automatic_promotion','evaluator_id','evaluator_version','completed_assessments','failed_assessments'];const cell=value=>{let text=value==null?'':String(value);if(typeof value==='string'&&/^\s*[=+@-]/.test(text))text="'"+text;return '"'+text.replace(/"/g,'""')+'"';};const scoped=new Set(['evaluator_id','evaluator_version','completed_assessments','failed_assessments']);const rows=packet.evaluators.length?packet.evaluators:[{}];const csv='\ufeff'+[columns.map(cell).join(','),...rows.map(row=>columns.map(key=>cell(scoped.has(key)?row[key]:packet[key])).join(','))].join('\r\n')+'\r\n';const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),link=node('a');link.href=url;link.download='callback-evaluator-attempts-'+project+'.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};}else body.append(node('p','Сервер не передал счётчики по версиям оценщиков.'));

  }catch(error){if(current())body.replaceChildren(node('p',error.message));}
}
$('callback-summary-open').onclick=()=>{$('callback-summary-dialog').showModal();return loadCallbackSummary();};
$('callback-summary-close').onclick=()=>$('callback-summary-dialog').close();
$('callback-summary-dialog').onclose=()=>{callbackSummaryEpoch++;};
$('callback-summary-refresh').onclick=()=>loadCallbackSummary();
for(const id of ['callback-summary-since','callback-summary-until'])$(id).oninput=()=>{callbackSummaryEpoch++;$('callback-summary-body').replaceChildren();};
function evaluatorReferenceRows(reference){
  if(reference==null)return [];
  if(typeof reference!=='object'||Array.isArray(reference)||Object.keys(reference).length!==2||typeof reference.id!=='string'||!/^[A-Za-z0-9._-]{1,100}$/.test(reference.id)||!Number.isSafeInteger(reference.version)||reference.version<1)return [node('p','Некорректные сведения об оценщике.')];
  return [node('p','Оценщик: '+reference.id+' · версия '+reference.version+' · идентификатор сообщён вашим кодом.')];
}
function evaluationSamplingRows(receipt){
  if(receipt==null)return [];
  if(typeof receipt!=='object'||Array.isArray(receipt)||Object.keys(receipt).length!==3||receipt.method!=='sha256_v1'||typeof receipt.sample_rate!=='number'||!Number.isFinite(receipt.sample_rate)||receipt.sample_rate<0||receipt.sample_rate>1||typeof receipt.selected!=='boolean')return [node('p','Некорректные сведения об отборе на оценку.')];
  return [node('p',(receipt.selected?'Выбран для оценки':'Оценка пропущена по отбору')+' · заданная доля: '+(receipt.sample_rate*100)+'%'+(receipt.selected?' · результат смотрите в дочернем шаге.':' · оценка качества отсутствует.'))];
}
function evaluationScoreRows(scores){
  if(scores==null)return [];
  if(typeof scores!=='object'||Array.isArray(scores)||!Object.keys(scores).length||Object.keys(scores).length>20||Object.entries(scores).some(([metric,value])=>!/^[A-Za-z0-9._-]{1,100}$/.test(metric)||typeof value!=='number'||!Number.isFinite(value)||value<0||value>1))return [node('p','Некорректные сообщённые оценки.')];
  const detail=node('details');detail.append(node('summary','Оценки callback · '+Object.keys(scores).length),node('p','Сообщены вашим кодом. Независимая проверка качества не выполнялась.'));for(const [metric,value] of Object.entries(scores).sort(([a],[b])=>a.localeCompare(b)))detail.append(node('p',metric+': '+value));return [detail];
}
function traceView(trace,relatedDepth=0,removed=false,reviewSpan=null,reviewerName=null){
  const detail=node('details'),header=node('summary',new Date(trace.started_ms).toLocaleString()+' · '+(evaluationLabels[trace.status]||trace.status));detail.append(header);
  if(trace.recovered_ms)detail.append(node('p','Вызов прерван предыдущим завершением Studio. Точное время завершения неизвестно.'));
  const depths=new Map(),labels={external_agent:'Внешний агент',external_model:'Внешняя модель',external_tool:'Внешний инструмент',turn:'Запуск',model:'Модель',tool:'Инструмент',swarm_member:'Участник',synthesis:'Синтез',critic:'Критик',compaction:'Сжатие контекста',experiment:'Проверка',evaluation_item:'Пример',remote_worker:'Удалённый участник',judge_batch:'Пакетная оценка',judge_item:'Пример оценки',llm_judge:'Оценка моделью'};
  for(const span of trace.spans){
    const depth=span.parent_id==null?0:Math.min(12,(depths.get(span.parent_id)||0)+1);depths.set(span.id,depth);
    const usage=span.usage||{},input=usage.prompt_tokens??usage.input_tokens,output=usage.completion_tokens??usage.output_tokens;
    const line=node('div',(labels[span.kind]||span.kind)+': '+span.name+' · '+(evaluationLabels[span.status]||span.status)+' · '+(span.duration_ms==null?(span.status==='running'?'выполняется':'длительность неизвестна'):span.duration_ms+' мс'));line.className='trace-span';line.style.paddingLeft=depth+'rem';if(span.provider_id)line.append(node('span',' · провайдер: '+span.provider_id));
    line.append(...guardrailReceiptRows(usage.guardrail_receipt),...evaluatorReferenceRows(usage.evaluator_ref),...evaluationSamplingRows(usage.evaluation_sampling),...evaluationScoreRows(usage.evaluation_scores));
    if(input!==undefined||output!==undefined)line.append(node('span',' · токены: '+(input??'?')+' / '+(output??'?')));
    if(['model','external_model','remote_worker'].includes(span.kind))line.append(node('span',typeof usage.cost==='number'?' · стоимость (провайдер): '+usage.cost:' · стоимость не сообщена'));
    if(!removed&&span.linked_trace_id&&relatedDepth<5){
      const open=node('button','Открыть вызов модели'),linked=node('div');
      open.onclick=async()=>{open.disabled=true;try{
        const child=await api('observability/traces/'+encodeURIComponent(span.linked_trace_id));
        if(child.project_id!==trace.project_id)throw new Error('Связанный вызов принадлежит другому проекту.');
        const view=traceView(child,relatedDepth+1);view.open=true;linked.replaceChildren(view);
      }catch(error){fail(error);open.disabled=false;}};
      line.append(open,linked);
    }
    detail.append(line);
  }
  if(removed)detail.append(node('p','В корзине. Исходные данные и отзывы сохранены.'));
  else detail.append(feedbackPanel(trace,reviewSpan,reviewerName),reviewQueueCreationPanel(trace));
  if(removed||trace.status!=='running'&&trace.spans.every(span=>span.status!=='running')){
    if(!removed){
      const includeFeedback=node('input');includeFeedback.type='checkbox';includeFeedback.checked=false;
      const label=node('label','Включить отзывы и исправленные ответы в экспорт');label.prepend(includeFeedback);
      const exportButton=node('button','Экспорт вызова JSON'),exportMessage=node('p');
      exportButton.onclick=async()=>{exportButton.disabled=true;try{
        if(trace.project_id!==$('project').value)throw new Error('Проект изменился.');
        const packet=await api('observability/traces/'+encodeURIComponent(trace.id)+'/export?include_feedback='+includeFeedback.checked);
        if(trace.project_id!==$('project').value||packet.trace.id!==trace.id||packet.trace.project_id!==trace.project_id)throw new Error('Экспорт относится к другому вызову или проекту.');
        const url=URL.createObjectURL(new Blob([JSON.stringify(packet,null,2)],{type:'application/json'})),link=node('a');
        link.href=url;link.download=trace.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
        exportMessage.textContent='Экспорт подготовлен. Проверьте загруженный файл.';
      }catch(error){exportMessage.textContent=error.message;}finally{exportButton.disabled=false;}};
      detail.append(label,exportButton,exportMessage);
    }
    const button=node('button',removed?'Восстановить вызов':'В корзину'),message=node('p');
    button.onclick=async()=>{button.disabled=true;try{
      if(trace.project_id!==$('project').value)throw new Error('Проект изменился. Откройте список заново.');
      await api('observability/traces/'+encodeURIComponent(trace.id)+(removed?'/restore':'/remove'),{});
      clearTraceSummary();
      detail.replaceChildren(node('summary',removed?'Вызов восстановлен':'Вызов перемещён в корзину'),node('p','Обновите список вызовов, чтобы увидеть актуальные записи.'));detail.open=true;
      const undo=node('button',removed?'Снова в корзину':'Восстановить вызов');
      undo.onclick=async()=>{undo.disabled=true;try{
        if(trace.project_id!==$('project').value)throw new Error('Проект изменился.');
        await api('observability/traces/'+encodeURIComponent(trace.id)+(removed?'/remove':'/restore'),{});
        clearTraceSummary();detail.replaceChildren(node('summary','Действие отменено'));detail.open=true;
      }catch(error){undo.disabled=false;fail(error);}};detail.append(undo);
    }catch(error){button.disabled=false;message.textContent=error.message;}};
    detail.append(button,message);
  }
  if(!removed&&trace.status==='completed'&&!trace.spans.some(span=>span.status==='running')){const capture=node('button','Передать текст для оценки');capture.onclick=()=>openQualityCapture(trace);detail.append(capture);}
  return detail;
}
let policyDraft=[];
function drawPolicyDraft(){const rows=policyDraft.map((rule,index)=>{const row=node('div',rule.id+' · '+rule.kind+' · '+JSON.stringify(rule.value)),remove=node('button','Убрать');remove.onclick=()=>{policyDraft.splice(index,1);drawPolicyDraft();};row.append(remove);return row;});$('policy-rule-draft').replaceChildren(...rows);$('policy-rule-create').disabled=!policyDraft.length;}
$('policy-rule-add').onclick=()=>{try{
  const id=$('policy-rule-id').value.trim(),kind=$('policy-rule-kind').value,raw=$('policy-rule-value').value;
  if(!/^[A-Za-z0-9._-]{1,80}$/.test(id)||policyDraft.some(rule=>rule.id===id))throw new Error('Укажите уникальное название латиницей, цифрами, точкой, дефисом или подчёркиванием.');
  if(policyDraft.length>=32)throw new Error('Не более 32 правил.');
  let value;if(kind==='json_valid')value=true;else if(kind==='min_bytes'||kind==='max_bytes'){if(!/^[0-9]+$/.test(raw.trim()))throw new Error('Укажите целое число от 0 до 64000.');value=Number(raw.trim());if(!Number.isSafeInteger(value)||value>64000)throw new Error('Укажите целое число от 0 до 64000.');}else if(kind==='forbidden_substrings'||kind==='required_substrings'){value=raw.split(/\r?\n/);if(value.length>100||value.some(fragment=>!fragment||new TextEncoder().encode(fragment).length>1000))throw new Error('Укажите 1–100 непустых фрагментов, до 1000 байт каждый.');}else throw new Error('Неизвестная проверка.');
  policyDraft.push({id,kind,value});drawPolicyDraft();$('policy-rule-message').replaceChildren();
}catch(error){$('policy-rule-message').replaceChildren(node('p',error.message));}};
$('policy-rule-create').onclick=async()=>{
  if(!policyDraft.length||!$('guardrail-policies-dialog').open)return;const epoch=policyCatalogEpoch,project=$('project').value,rules=JSON.parse(JSON.stringify(policyDraft));$('policy-rule-create').disabled=true;
  const current=()=>epoch===policyCatalogEpoch&&project===$('project').value&&$('guardrail-policies-dialog').open;
  try{const receipt=await api('guardrail-policies/create',{rules});if(!current())return;if(receipt.provider_calls!==0||receipt.policy?.kind!=='local_guardrail_policy'||receipt.policy?.schema_version!==1||!/^[a-f0-9]{64}$/.test(receipt.policy.policy_sha256)||JSON.stringify(receipt.policy.rules)!==JSON.stringify(rules))throw new Error('Несовместимая созданная политика.');notify('Новая политика сохранена');if(JSON.stringify(policyDraft)!==JSON.stringify(rules))return;policyDraft=[];drawPolicyDraft();$('guardrail-policies-open').click();}catch(error){if(current())$('policy-rule-message').replaceChildren(node('p',error.message));}finally{if(current())$('policy-rule-create').disabled=!policyDraft.length;}
};
let policyCatalogEpoch=0,policyImportEpoch=0,policyImport=null;
function clearPolicyImport(){policyDraft=[];drawPolicyDraft();$('policy-rule-message').replaceChildren();policyImportEpoch++;policyImport=null;$('guardrail-policy-file').value='';$('guardrail-policy-preview').replaceChildren();$('guardrail-policy-save').disabled=true;}
$('guardrail-policy-file').onchange=async()=>{
  const epoch=++policyImportEpoch,catalog=policyCatalogEpoch,project=$('project').value,file=$('guardrail-policy-file').files[0],preview=$('guardrail-policy-preview');policyImport=null;$('guardrail-policy-save').disabled=true;preview.replaceChildren();
  const current=()=>epoch===policyImportEpoch&&catalog===policyCatalogEpoch&&project===$('project').value&&$('guardrail-policies-dialog').open;
  try{if(!file)return;if(file.size>128*1024)throw new Error('Файл правил превышает 128 КиБ.');const raw=await file.text();if(!current())return;const manifest=JSON.parse(raw);
    if(manifest.kind!=='local_guardrail_policy'||manifest.schema_version!==1||!/^[a-f0-9]{64}$/.test(manifest.policy_sha256)||!Array.isArray(manifest.rules)||manifest.rules.length<1||manifest.rules.length>32)throw new Error('Некорректный файл правил.');
    policyImport={raw,manifest,project,catalog};preview.replaceChildren(node('p','Правил: '+manifest.rules.length+' · отпечаток: '+manifest.policy_sha256),node('pre',JSON.stringify(manifest.rules,null,2)),node('p','При сохранении сервер проверит отпечаток и полный формат правил.'));$('guardrail-policy-save').disabled=false;
  }catch(error){if(current())preview.replaceChildren(node('p',error.message));}
};
$('guardrail-policy-save').onclick=async()=>{
  const imported=policyImport,epoch=policyImportEpoch;if(!imported||imported.project!==$('project').value||imported.catalog!==policyCatalogEpoch||!$('guardrail-policies-dialog').open)return;
  const current=()=>epoch===policyImportEpoch&&imported===policyImport&&imported.catalog===policyCatalogEpoch&&imported.project===$('project').value&&$('guardrail-policies-dialog').open;
  $('guardrail-policy-save').disabled=true;
  try{const response=await fetch('/api/guardrail-policies',{method:'POST',headers:{'Content-Type':'application/json','X-Allpaka-Client':'studio'},body:imported.raw});const receipt=await response.json();if(!current())return;if(!response.ok)throw new Error(receipt.error||'Не удалось сохранить правила.');if(receipt.provider_calls!==0||receipt.policy?.policy_sha256!==imported.manifest.policy_sha256)throw new Error('Несовместимый ответ сохранения.');$('guardrail-policy-preview').replaceChildren(node('p','Правила сохранены: '+receipt.policy.policy_sha256));policyImport=null;notify('Правила сохранены');$('guardrail-policies-open').click();
  }catch(error){if(current()){$('guardrail-policy-preview').append(node('p',error.message));$('guardrail-policy-save').disabled=false;}}
};
$('guardrail-policies-close').onclick=()=>{policyCatalogEpoch++;clearPolicyImport();$('guardrail-policies-dialog').close();};
$('guardrail-policies-open').onclick=()=>{
  clearPolicyImport();const epoch=++policyCatalogEpoch,project=$('project').value,dialog=$('guardrail-policies-dialog'),panel=$('guardrail-policies-result');dialog.showModal();
  const current=()=>epoch===policyCatalogEpoch&&dialog.open&&project===$('project').value;
  let pageEpoch=0;
  const show=async(offset=0)=>{const page=++pageEpoch;panel.replaceChildren(node('p','Загрузка правил…'));try{
    const receipt=await api('guardrail-policies?offset='+offset+'&limit=20');if(!current()||page!==pageEpoch)return;
    if(receipt.provider_calls!==0||receipt.offset!==offset||receipt.limit!==20||!Number.isSafeInteger(receipt.total)||receipt.total<0||!Array.isArray(receipt.policies)||receipt.policies.length>20)throw new Error('Некорректный каталог правил.');
    const rows=[node('p','Сохранено политик: '+receipt.total)];
    for(const policy of receipt.policies){
      if(!/^[a-f0-9]{64}$/.test(policy.policy_sha256)||!Number.isSafeInteger(policy.rule_count)||policy.rule_count<1||policy.rule_count>32||policy.schema_version!==1)throw new Error('Некорректная запись правил.');
      const row=node('details'),content=node('div'),open=node('button','Показать правила'),traces=node('button','Найти вызовы с этими правилами');row.append(node('summary',policy.policy_sha256+' · правил: '+policy.rule_count));
      open.onclick=async()=>{open.disabled=true;try{const saved=await api('guardrail-policies/'+policy.policy_sha256);if(!current()||page!==pageEpoch)return;if(saved.provider_calls!==0||saved.policy?.policy_sha256!==policy.policy_sha256||saved.policy?.kind!=='local_guardrail_policy'||saved.policy?.schema_version!==1||!Array.isArray(saved.policy.rules)||saved.policy.rules.length!==policy.rule_count)throw new Error('Несовместимые правила.');content.replaceChildren(node('pre',JSON.stringify(saved.policy.rules,null,2)));}catch(error){if(current()&&page===pageEpoch)content.replaceChildren(node('p',error.message));}finally{if(current()&&page===pageEpoch)open.disabled=false;}};
      const sample=node('textarea'),check=node('button','Проверить текст'),checked=node('div');sample.setAttribute('aria-label','Текст для проверки');sample.placeholder='Текст обрабатывается проверкой и не сохраняется в трассах';
      let checkEpoch=0;
      check.onclick=async()=>{const request=++checkEpoch,text=sample.value||'';check.disabled=true;try{if(new TextEncoder().encode(text).length>64000)throw new Error('Не более 64000 байт текста.');const receipt=await api('guardrail-policies/'+policy.policy_sha256+'/check',{text,stage:'input',action:'observe'});if(!current()||page!==pageEpoch||request!==checkEpoch)return;if(receipt.kind!=='local_guardrail'||receipt.policy_sha256!==policy.policy_sha256||receipt.provider_calls!==0||receipt.content_captured!==false||receipt.stage!=='input'||receipt.action!=='observe'||typeof receipt.passed!=='boolean'||!Array.isArray(receipt.rules))throw new Error('Несовместимый результат проверки.');checked.replaceChildren(node('p',receipt.passed?'Все правила выполнены':'Есть нарушения'),...receipt.rules.map(rule=>node('p',rule.rule_id+': '+(rule.passed?'успешно':'нарушение'))));}catch(error){if(current()&&page===pageEpoch&&request===checkEpoch)checked.replaceChildren(node('p',error.message));}finally{if(current()&&page===pageEpoch&&request===checkEpoch)check.disabled=false;}};
      sample.oninput=()=>{checkEpoch++;check.disabled=false;checked.replaceChildren();};
      traces.onclick=()=>{if(!current()||page!==pageEpoch)return;$('trace-guardrail-policy').value=policy.policy_sha256;dialog.close();policyCatalogEpoch++;$('trace-project').click();};const input=node('button','Для входа чата'),output=node('button','Для ответа чата');for(const [button,field] of [[input,'chat-guardrails-input'],[output,'chat-guardrails-output']])button.onclick=()=>{if(!current()||page!==pageEpoch)return;$(field).value=policy.policy_sha256;notify('Политика выбрана. Включите проверки в настройках чата.');};row.append(open,traces,content,sample,check,checked,input,output);rows.push(row);
    }
    const previous=node('button','Предыдущие'),next=node('button','Следующие');previous.disabled=offset===0;next.disabled=offset+20>=receipt.total;previous.onclick=()=>show(Math.max(0,offset-20));next.onclick=()=>show(offset+20);rows.push(previous,next);panel.replaceChildren(...rows);
  }catch(error){if(current()&&page===pageEpoch)panel.replaceChildren(node('p',error.message));}};
  return show();
};
function firstTextRows(latency){
  if(!latency)return [node('p','Первый текст: данные не сообщены.')];
  const ms=value=>value==null?'неизвестно':value+' мс';
  return [node('p','Задержка первого текста: P50 '+ms(latency.p50_ms)+' · P95 '+ms(latency.p95_ms)+' · минимум '+ms(latency.min_ms)+' · максимум '+ms(latency.max_ms)),node('p','Первый текст · измерений: '+latency.known_calls+' · без измерения: '+latency.unknown_calls)];
}
function timeSeriesView(result,metric,isCurrent=()=>true){
  const tokenFields={input_tokens:['input_tokens','input_tokens_unknown_calls','входные'],output_tokens:['output_tokens','output_tokens_unknown_calls','выходные'],cache_creation:['cache_creation_input_tokens','cache_creation_unknown_calls','создания кэша'],cache_read:['cache_read_input_tokens','cache_read_unknown_calls','чтения кэша']};
  const token=Object.hasOwn(tokenFields,metric)?tokenFields[metric]:null;
  const currency=$('trace-series-currency').value.trim();
  if(metric==='cost'&&!/^[A-Z]{3}$/.test(currency))throw new Error('Укажите технический код валюты, например USD.');
  if(metric==='cost'&&result.buckets.some(row=>{const costs=row.model_usage.reported_cost_by_currency;return !costs||typeof costs!=='object'||Array.isArray(costs)||Object.entries(costs).some(([code,value])=>! /^[A-Z]{3}$/.test(code)||typeof value!=='number'||!Number.isFinite(value)||value<0);}))throw new Error('Некорректные суммы по валютам.');
  const values=result.buckets.map(row=>metric==='traces'?row.trace_count:metric==='models'?row.model_calls:metric==='errors'?(row.trace_statuses.failed??0):metric==='guardrails'?row.guardrails.violations:token?row.model_usage[token[0]]:metric==='cost'?(row.model_usage.reported_cost_by_currency[currency]??0):NaN);
  if(values.some(value=>(metric==='cost'?!Number.isFinite(value):!Number.isSafeInteger(value))||value<0))throw new Error('Некорректные показатели графика.');
  if(token&&result.buckets.some(row=>!Number.isSafeInteger(row.model_usage[token[1]])||row.model_usage[token[1]]<0))throw new Error('Некорректные сведения о неизвестных токенах.');
  if(metric==='cost'&&result.buckets.some(row=>['cost_unknown_calls','cost_currency_unknown_calls'].some(field=>!Number.isSafeInteger(row.model_usage[field])||row.model_usage[field]<0)))throw new Error('Некорректные сведения о стоимости.');
  const view=node('div'),bars=node('div',undefined,'trace-series-bars'),detail=node('div'),maximum=Math.max(1,...values);
  let selectionEpoch=0;
  const select=index=>{const row=result.buckets[index],selection=++selectionEpoch;detail.replaceChildren(node('h4',new Date(row.start_ms).toLocaleString()+' — '+new Date(row.end_exclusive_ms-1).toLocaleString()),node('p','Вызовов: '+row.trace_count+' · вызовов моделей: '+row.model_calls+' · ошибок: '+(row.trace_statuses.failed??0)),node('p','Нарушений проверок: '+row.guardrails.violations+' · блокировок: '+row.guardrails.blocked),...firstTextRows(row.model_usage.first_text));
    if(token)detail.append(node('p','Сообщённые токены '+token[2]+': '+row.model_usage[token[0]]+' · без данных: '+row.model_usage[token[1]]+' вызовов. Столбец показывает только сообщённые токены.'));
    if(metric==='cost'){for(const field of ['cost_unknown_calls','cost_currency_unknown_calls'])if(!Number.isSafeInteger(row.model_usage[field])||row.model_usage[field]<0)throw new Error('Некорректные сведения о стоимости.');detail.append(node('p','Сообщённая стоимость '+currency+': '+values[index]+' · без стоимости: '+row.model_usage.cost_unknown_calls+' · без валюты: '+row.model_usage.cost_currency_unknown_calls),node('p','Валюты не суммируются. Другие сообщённые валюты: '+(Object.keys(row.model_usage.reported_cost_by_currency).filter(code=>code!==currency).join(', ')||'нет')+'. Столбцы показывают только сообщённые суммы.'));}
    const open=node('button','Открыть вызовы интервала'),list=node('div');open.type='button';open.disabled=row.trace_count===0||(metric==='errors'&&result.status&&result.status!=='failed');
    const current=()=>selection===selectionEpoch&&isCurrent();
    let pageEpoch=0;
    const show=async(offset=0)=>{const pageRequest=++pageEpoch;try{
      const params=new URLSearchParams({project_id:result.project_id,since_ms:String(row.start_ms),until_ms:String(row.end_exclusive_ms-1),offset:String(offset),limit:'20'});
      if(result.session_id)params.set('session_id',result.session_id);if(result.status)params.set('status',result.status);if(metric==='errors')params.set('status','failed');if(metric==='guardrails')params.set('guardrail','failed');
      const page=await api('observability/traces?'+params);if(!current()||pageRequest!==pageEpoch)return;
      if(!Array.isArray(page.traces)||page.traces.length>20||!Number.isSafeInteger(page.total)||page.total<0||page.traces.length>page.total||page.traces.some(trace=>trace.project_id!==result.project_id||!Number.isSafeInteger(trace.started_ms)||trace.started_ms<row.start_ms||trace.started_ms>=row.end_exclusive_ms||(result.session_id&&trace.session_id!==result.session_id)||((metric==='errors'?'failed':result.status)&&trace.status!==(metric==='errors'?'failed':result.status))))throw new Error('Получены вызовы вне выбранного интервала.');
      const controls=node('div'),previous=node('button','Предыдущие'),next=node('button','Следующие');previous.disabled=offset===0;next.disabled=offset+20>=page.total;
      const load=offset=>{if(!current()||pageRequest!==pageEpoch)return;return show(offset);};previous.onclick=()=>{if(!previous.disabled)return load(Math.max(0,offset-20));};next.onclick=()=>{if(!next.disabled)return load(offset+20);};controls.append(previous,next);
      list.replaceChildren(node('h4','Найдено вызовов: '+page.total),...page.traces.map(trace=>traceView(trace)),controls);
    }catch(error){if(current()&&pageRequest===pageEpoch)list.replaceChildren(node('p',error.message));}};
    open.onclick=async()=>{if(open.disabled||!current())return;open.disabled=true;try{await show();}catch(error){if(current())list.replaceChildren(node('p',error.message));}finally{if(current())open.disabled=false;}};
    detail.append(open,list);
  };
  result.buckets.forEach((row,index)=>{const button=node('button',undefined,'trace-series-bucket'),bar=node('span',undefined,'trace-series-bar');
    button.type='button';button.title=new Date(row.start_ms).toLocaleString()+': '+values[index];button.setAttribute('aria-label',button.title);
    bar.style.height=Math.max(2,160*values[index]/maximum)+'px';button.append(node('span',String(values[index])),bar,node('span',new Date(row.start_ms).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})));button.onclick=()=>select(index);bars.append(button);
  });
  view.append(node('p','Интервалов: '+result.bucket_count+' · выберите столбец для подробностей.'),bars,detail);if(result.buckets.length)select(0);return view;
}
let traceSeriesEpoch=0,traceSeriesReceipt=null;
function clearTraceSeries(){traceSeriesEpoch++;traceSeriesReceipt=null;$('trace-series-export').disabled=true;$('trace-series-result').replaceChildren();}
$('trace-series-interval').onchange=clearTraceSeries;
$('trace-series-metric').onchange=()=>{try{const receipt=traceSeriesReceipt,epoch=traceSeriesEpoch;if(receipt&&receipt.project_id===$('project').value)$('trace-series-result').replaceChildren(timeSeriesView(receipt,$('trace-series-metric').value,()=>receipt===traceSeriesReceipt&&epoch===traceSeriesEpoch&&receipt.project_id===$('project').value&&$('trace-summary-dialog').open));}catch(error){$('trace-series-result').replaceChildren(node('p',error.message));}};
$('trace-series-currency').onchange=()=>{try{$('trace-series-metric').onchange();}catch(error){$('trace-series-result').replaceChildren(node('p',error.message));}};
$('trace-series-load').onclick=async()=>{
  clearTraceSeries();const epoch=traceSeriesEpoch,project=$('project').value,session=$('trace-summary-scope').value==='session'?traceSummarySession:null;
  const status=$('trace-summary-status').value||null;
  const current=()=>epoch===traceSeriesEpoch&&$('trace-summary-dialog').open&&project===$('project').value;
  try{
    if($('trace-summary-scope').value==='session'&&!session)throw new Error('Сначала откройте разговор.');
    const until=$('trace-summary-until').value?new Date($('trace-summary-until').value).getTime():Date.now();
    const since=$('trace-summary-since').value?new Date($('trace-summary-since').value).getTime():Math.max(0,until-86400000);
    const interval=Number($('trace-series-interval').value);
    if(![900000,3600000,86400000].includes(interval)||!Number.isSafeInteger(since)||!Number.isSafeInteger(until)||since<0||since>until||(until-since)/interval>=500)throw new Error('Выберите корректный период, не более 500 интервалов.');
    const params=new URLSearchParams({project_id:project,since_ms:String(since),until_ms:String(until),bucket_ms:String(interval)});if(session)params.set('session_id',session);
    if(status)params.set('status',status);
    const result=await api('observability/time-series?'+params);if(!current())return;
    if(result.kind!=='trace_time_series'||result.schema_version!==1||result.provider_calls!==0||result.project_id!==project||result.session_id!==session||(result.status??null)!==status||result.since_ms!==since||result.until_ms!==until||result.bucket_ms!==interval||!Array.isArray(result.buckets)||result.bucket_count!==result.buckets.length||result.bucket_count!==Math.floor((until-since)/interval)+1)throw new Error('Получен несовместимый график активности.');
    if(result.buckets.some((row,index)=>row.start_ms!==since+index*interval||row.end_exclusive_ms!==Math.min(since+(index+1)*interval,until+1)))throw new Error('Некорректные интервалы графика.');
    const view=timeSeriesView(result,$('trace-series-metric').value,()=>current()&&traceSeriesReceipt===result);traceSeriesReceipt=result;$('trace-series-export').disabled=false;$('trace-series-result').replaceChildren(view);
  }catch(error){if(current())$('trace-series-result').replaceChildren(node('p',error.message));}
};
$('trace-series-export').onclick=()=>{
  const receipt=traceSeriesReceipt;if(!receipt||receipt.project_id!==$('project').value||!$('trace-summary-dialog').open)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(receipt,null,2)],{type:'application/json'})),link=node('a');link.href=url;link.download='allpaka-activity-'+receipt.since_ms+'-'+receipt.until_ms+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
let traceSummaryEpoch=0,traceSummaryProject=null,traceSummarySession=null,traceSummaryReceipt=null;
function clearTraceSummary(){clearTraceSeries();traceSummaryEpoch++;traceSummaryReceipt=null;$('trace-summary-export').disabled=true;$('trace-summary-result').replaceChildren();}
for(const id of ['trace-summary-scope','trace-summary-status','trace-summary-since','trace-summary-until'])$(id).onchange=clearTraceSummary;
$('trace-summary-open').onclick=()=>{
  clearTraceSummary();traceSummaryProject=$('project').value;traceSummarySession=active;
  $('trace-summary-scope').value='project';$('trace-summary-scope').querySelector('[value="session"]').disabled=!active;
  $('trace-summary-dialog').showModal();$('trace-summary-refresh').click();
};
$('trace-summary-close').onclick=()=>{$('trace-summary-dialog').close();clearTraceSummary();};
async function loadConversationTraces(container,scope,isCurrent){
  let pageEpoch=0;
  const show=async(offset=0)=>{
    if(!isCurrent())return;
    const page=++pageEpoch,current=()=>isCurrent()&&page===pageEpoch;
    container.replaceChildren(node('p','Загрузка вызовов…'));
    try{
      const params=new URLSearchParams({project_id:scope.project_id,session_id:scope.session_id,offset:String(offset),limit:'20'});
      for(const key of ['since_ms','until_ms','status'])if(scope[key]!=null)params.set(key,scope[key]);
      const result=await api('observability/traces?'+params);
      if(!current())return;
      if(!Array.isArray(result.traces)||!Number.isSafeInteger(result.total)||result.total<0||result.traces.length>20||result.traces.some(trace=>trace.project_id!==scope.project_id||trace.session_id!==scope.session_id||scope.status!=null&&trace.status!==scope.status||scope.since_ms!=null&&trace.started_ms<Number(scope.since_ms)||scope.until_ms!=null&&trace.started_ms>Number(scope.until_ms)))throw new Error('Полученные вызовы не соответствуют разговору и периоду.');
      const rows=[node('p','Вызовов: '+result.total),...result.traces.map(trace=>traceView(trace))];
      const previous=node('button','Предыдущие вызовы'),next=node('button','Следующие вызовы');previous.disabled=offset===0;next.disabled=offset+result.traces.length>=result.total;
      previous.onclick=()=>{if(current()&&!previous.disabled)show(Math.max(0,offset-20));};next.onclick=()=>{if(current()&&!next.disabled)show(offset+20);};rows.push(previous,next);container.replaceChildren(...rows);
    }catch(error){if(current()){const retry=node('button','Повторить загрузку');retry.onclick=()=>{if(current())show(offset);};container.replaceChildren(node('p',error.message),retry);}}
  };
  await show();
}
async function loadTraceSummary(conversationOffset=0){
  clearTraceSummary();const epoch=traceSummaryEpoch,project=traceSummaryProject,session=$('trace-summary-scope').value==='session'?traceSummarySession:null,button=$('trace-summary-refresh');
  button.disabled=true;
  try{
    if(!project||project!==$('project').value)throw new Error('Проект изменился. Откройте сводку заново.');
    if($('trace-summary-scope').value==='session'&&!session)throw new Error('Сначала откройте разговор.');
    const params=new URLSearchParams({project_id:project});if(session)params.set('session_id',session);
    for(const [id,key] of [['trace-summary-since','since_ms'],['trace-summary-until','until_ms']])if($(id).value){const value=new Date($(id).value).getTime();if(!Number.isFinite(value)||value<0)throw new Error('Некорректная дата.');params.set(key,String(value));}
    if($('trace-summary-status').value)params.set('status',$('trace-summary-status').value);
    params.set('conversation_offset',String(conversationOffset));params.set('conversation_limit','100');
    const result=await api('observability/summary?'+params);
    if(epoch!==traceSummaryEpoch||!$('trace-summary-dialog').open||project!==$('project').value)return;
    traceSummaryReceipt={project_id:project,session_id:session,fetched_ms:Date.now(),summary:result};$('trace-summary-export').disabled=false;
    const rows=[node('p','Сохранённых трасс: '+result.trace_count+' · вызовов модели: '+result.model_calls),node('p','Сообщённые входные токены: '+result.input_tokens+' · без данных: '+result.input_tokens_unknown_calls+' вызовов'),node('p','Сообщённые выходные токены: '+result.output_tokens+' · без данных: '+result.output_tokens_unknown_calls+' вызовов')];
    rows.push(node('p','Токены создания кэша: '+(result.cache_creation_input_tokens ?? 'нет данных')+' · без данных: '+(result.cache_creation_unknown_calls ?? 'нет данных')+' вызовов'),node('p','Токены чтения кэша: '+(result.cache_read_input_tokens ?? 'нет данных')+' · без данных: '+(result.cache_read_unknown_calls ?? 'нет данных')+' вызовов'));
    rows.push(...firstTextRows(result.first_text),node('p','Задержка первого текста измеряется приложением и включает соединение и паузы чтения. Серверное время генерации первого токена здесь неизвестно.'));
    if(result.guardrails){const g=result.guardrails;rows.push(node('h3','Проверки входа и ответа'),node('p','Проверок: '+g.checks+' · успешно: '+g.passed+' · нарушений: '+g.violations+' · блокировок: '+g.blocked),node('p','Незавершённые или противоречивые: '+g.incomplete_or_inconsistent+' · вход: '+g.input_checks+' · ответ: '+g.output_checks));
      for(const policy of g.policies||[])rows.push(node('p','Правила '+policy.policy_sha256+' · проверок: '+policy.checks+' · нарушений: '+policy.violations+' · блокировок: '+policy.blocked));
      if(g.policy_groups_truncated)rows.push(node('p','Показаны первые '+g.policy_groups_limit+' групп правил; общие суммы включают все проверки.'));
    }
    for(const [status,count] of Object.entries(result.trace_statuses))rows.push(node('p',(evaluationLabels[status]||status)+': '+count));
    if(result.conversations?.length){
      rows.push(node('h3','По разговорам'));
      for(const conversation of result.conversations){
        const usage=conversation.model_usage,detail=node('details');
        detail.append(node('summary',conversation.session_id+' · трасс: '+conversation.trace_count+' · вызовов моделей: '+usage.calls));
        detail.append(node('p','Внешних вызовов моделей: '+(conversation.external_model_calls??0)));
        detail.append(node('p','Входные токены: '+usage.input_tokens+' · без данных: '+usage.input_tokens_unknown_calls),node('p','Выходные токены: '+usage.output_tokens+' · без данных: '+usage.output_tokens_unknown_calls));
        for(const [status,count] of Object.entries(conversation.trace_statuses))detail.append(node('p',(evaluationLabels[status]||status)+': '+count));
        for(const [currency,value] of Object.entries(usage.reported_cost_by_currency))detail.append(node('p','Стоимость '+currency+': '+value));
        detail.append(node('p','Без стоимости: '+usage.cost_unknown_calls+' · без валюты: '+usage.cost_currency_unknown_calls));
        const open=node('button','Показать вызовы разговора'),traces=node('div');
        open.onclick=()=>{if(epoch!==traceSummaryEpoch||project!==$('project').value||!$('trace-summary-dialog').open)return;open.disabled=true;loadConversationTraces(traces,{project_id:project,session_id:conversation.session_id,status:params.get('status'),since_ms:params.get('since_ms'),until_ms:params.get('until_ms')},()=>epoch===traceSummaryEpoch&&project===$('project').value&&$('trace-summary-dialog').open);};
        detail.append(open,traces);rows.push(detail);
      }
      if(result.conversations_truncated)rows.push(node('p','Показаны разговоры '+(result.conversation_offset+1)+'–'+(result.conversation_offset+result.conversations.length)+' из '+result.conversation_count+'. Общие итоги включают все разговоры.'));
    }
    if((result.conversation_offset||0)>0||result.conversation_has_more){
      const pager=node('div'),previous=node('button','Предыдущие разговоры'),next=node('button','Следующие разговоры');
      previous.disabled=!(result.conversation_offset>0);next.disabled=!result.conversation_has_more;
      const move=offset=>{if(epoch!==traceSummaryEpoch||project!==$('project').value||!$('trace-summary-dialog').open)return;loadTraceSummary(offset);};
      previous.onclick=()=>{if(!previous.disabled)move(Math.max(0,result.conversation_offset-result.conversation_limit));};
      next.onclick=()=>{if(!next.disabled)move(result.conversation_offset+result.conversation_limit);};
      pager.append(previous,next);rows.push(pager);
    }
    rows.push(node('h3','Сообщённая стоимость'));
    const costs=Object.entries(result.reported_cost_by_currency);
    if(!costs.length)rows.push(node('p','Стоимость с указанной валютой не сообщена.'));
    for(const [currency,value] of costs)rows.push(node('p',currency+': '+value));
    rows.push(node('p','Стоимость не сообщена: '+result.cost_unknown_calls+' вызовов. Валюта не сообщена: '+result.cost_currency_unknown_calls+' значений; они не входят в денежный итог.'));
    if(result.models?.length){
      rows.push(node('h3','По моделям'));
      for(const model of result.models){
        const detail=node('details');detail.append(node('summary',model.model+' · '+(model.provider_id||'Провайдер неизвестен')+' · '+(model.source==='external'?'Внешние данные':'Studio')+' · вызовов: '+model.calls));
        detail.append(node('p','Входные токены: '+model.input_tokens+' · без данных: '+model.input_tokens_unknown_calls),node('p','Выходные токены: '+model.output_tokens+' · без данных: '+model.output_tokens_unknown_calls));
        detail.append(node('p','Создание кэша: '+(model.cache_creation_input_tokens ?? 'нет данных')+' · без данных: '+(model.cache_creation_unknown_calls ?? 'нет данных')),node('p','Чтение кэша: '+(model.cache_read_input_tokens ?? 'нет данных')+' · без данных: '+(model.cache_read_unknown_calls ?? 'нет данных')));
        detail.append(...firstTextRows(model.first_text));
        for(const [status,count] of Object.entries(model.statuses))detail.append(node('p',(evaluationLabels[status]||status)+': '+count));
        const duration=value=>value==null?'неизвестно':value+' мс';
        detail.append(node('p','Длительность вызова: P50 '+duration(model.duration_p50_ms)+' · P95 '+duration(model.duration_p95_ms)),node('p','Измерений: '+model.duration_known_calls+' · без длительности: '+model.duration_unknown_calls));
        for(const [currency,value] of Object.entries(model.reported_cost_by_currency))detail.append(node('p','Стоимость '+currency+': '+value));
        detail.append(node('p','Без стоимости: '+model.cost_unknown_calls+' · без валюты: '+model.cost_currency_unknown_calls));
        if(model.source==='external')detail.append(node('p','Время и расход сообщены внешним приложением.'));
        rows.push(detail);
      }
      rows.push(node('p','P50 и P95 рассчитаны по известной длительности всего вызова, включая ошибки. Время до первого токена здесь не измеряется.'));
    }
    if(result.reported_cost_examples.length){const examples=node('details');examples.append(node('summary','Отдельные расходы · до '+result.cost_examples_limit+' примеров'));
      for(const item of result.reported_cost_examples){const row=node('div',item.model+' · '+item.value+' · '+(item.currency||'валюта неизвестна')),open=node('button','Открыть вызов'),content=node('div');
        open.onclick=async()=>{open.disabled=true;try{const trace=await api('observability/traces/'+encodeURIComponent(item.trace_id));if(epoch!==traceSummaryEpoch)return;if(trace.project_id!==project)throw new Error('Вызов принадлежит другому проекту.');const view=traceView(trace);view.open=true;content.replaceChildren(view);}catch(error){content.replaceChildren(node('p',error.message));open.disabled=false;}};
        row.append(open,content);examples.append(row);
      }rows.push(examples);
    }
    $('trace-summary-result').replaceChildren(...rows);
  }catch(error){if(epoch===traceSummaryEpoch)$('trace-summary-result').replaceChildren(node('p',error.message));}finally{if(epoch===traceSummaryEpoch)button.disabled=false;}
};
$('trace-summary-refresh').onclick=()=>loadTraceSummary(0);
$('trace-summary-export').onclick=()=>{
  if(!traceSummaryReceipt)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(traceSummaryReceipt,null,2)],{type:'application/json'})),link=node('a');link.href=url;link.download='allpaka-summary-'+traceSummaryReceipt.fetched_ms+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
for(const id of ['trace-status','trace-since','trace-until'])$(id).onchange=()=>$('trace-project').click();
$('trace-guardrail').onchange=()=>$('trace-project').click();
$('trace-guardrail-policy').onchange=()=>$('trace-project').click();
let traceBulkEpoch=0;let traceCatalogEpoch=0;
function clearTraceBulk(){traceBulkEpoch++;traceCatalogEpoch++;}
$('trace-bulk').onclick=async()=>{traceCatalogEpoch++;
  const epoch=++traceBulkEpoch,project=$('project').value,panel=$('trace-results'),selected=new Set();
  const current=()=>epoch===traceBulkEpoch&&project===$('project').value;
  const count=node('p'),message=node('p'),exportButton=node('button','Скачать выбранные вызовы JSON'),clear=node('button','Снять выбор');
  const feedback=node('input');feedback.type='checkbox';feedback.checked=false;
  const label=node('label','Включить отзывы и исправленные ответы');label.prepend(feedback);
  let busy=false;
  const update=()=>{count.textContent='Выбрано: '+selected.size+' из 100';exportButton.disabled=busy||!selected.size;};
  const boxes=new Map();
  clear.onclick=()=>{if(!current())return;selected.clear();for(const box of boxes.values())box.checked=false;message.textContent='';update();};
  exportButton.onclick=async()=>{
    if(!current())return;
    const ids=[...selected],includeFeedback=feedback.checked;if(!ids.length)return;
    busy=true;update();message.textContent='Подготовка экспорта…';
    try{
      const packet=await api('observability/trace-exports',{project_id:project,trace_ids:ids,include_feedback:includeFeedback});
      if(!current())return;
      if(JSON.stringify([...selected])!==JSON.stringify(ids)||feedback.checked!==includeFeedback)throw new Error('Выбор изменился. Повторите экспорт.');
      if(packet.kind!=='trace_export_batch'||packet.schema_version!==1||packet.project_id!==project||packet.trace_count!==ids.length||!Array.isArray(packet.traces)||packet.traces.length!==ids.length||packet.provider_calls!==0||packet.privacy?.feedback_included!==includeFeedback||packet.traces.some((item,index)=>item.trace?.id!==ids[index]||item.trace?.project_id!==project||!includeFeedback&&item.feedback!==null))throw new Error('Ответ экспорта не соответствует выбранным вызовам.');
      const url=URL.createObjectURL(new Blob([JSON.stringify(packet,null,2)],{type:'application/json'})),link=node('a');
      link.href=url;link.download='allpaka-traces-'+Date.now()+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      message.textContent='Экспорт подготовлен. Проверьте загруженный файл.';
    }catch(error){if(current())message.textContent=error.message;}finally{busy=false;if(current())update();}
  };
  const show=async(offset=0)=>{
    const result=await api('observability/traces?project_id='+encodeURIComponent(project)+'&offset='+offset+'&limit=20');
    if(!current())return;
    if(result.traces.some(trace=>trace.project_id!==project))throw new Error('Список содержит вызовы другого проекта.');
    boxes.clear();
    const rows=result.traces.map(trace=>{
      const row=node('div'),box=node('input');box.type='checkbox';box.checked=selected.has(trace.id);
      box.disabled=trace.status==='running'||trace.spans.some(span=>span.status==='running');
      const choose=node('label','Выбрать '+trace.id);choose.prepend(box);boxes.set(trace.id,box);
      box.onchange=()=>{if(!current())return;if(box.checked){if(selected.size>=100){box.checked=false;message.textContent='Можно выбрать до 100 вызовов.';return;}selected.add(trace.id);}else selected.delete(trace.id);update();};
      row.append(choose,traceView(trace));return row;
    });
    const controls=node('div'),previous=node('button','Предыдущие вызовы'),next=node('button','Следующие вызовы');
    previous.disabled=offset===0;next.disabled=offset+20>=result.total;
    previous.onclick=()=>show(Math.max(0,offset-20)).catch(error=>{if(current())message.textContent=error.message;});next.onclick=()=>show(offset+20).catch(error=>{if(current())message.textContent=error.message;});controls.append(previous,next);
    panel.replaceChildren(node('h3','Экспорт выбранных вызовов'),count,label,exportButton,clear,message,...rows,controls);update();
  };
  try{await show();}catch(error){if(current())panel.replaceChildren(node('p',error.message));}
};
$('trace-project').onclick=async()=>{clearTraceBulk();
  const project=$('project').value,filter=$('trace-guardrail').value,hash=$('trace-guardrail-policy').value.trim(),epoch=traceCatalogEpoch,panel=$('trace-results');
  const status=$('trace-status').value,since=$('trace-since').value,until=$('trace-until').value;
  const current=()=>epoch===traceCatalogEpoch&&status===$('trace-status').value&&since===$('trace-since').value&&until===$('trace-until').value&&project===$('project').value&&filter===$('trace-guardrail').value&&hash===$('trace-guardrail-policy').value.trim();
  try{
    if(!['','any','failed','blocked'].includes(filter))throw new Error('Некорректный фильтр проверок.');
    if(hash&&!/^[a-f0-9]{64}$/.test(hash))throw new Error('Хеш правил должен содержать 64 символа: цифры и a–f.');
    if(status&&!/^[a-z_]{1,40}$/.test(status))throw new Error('Некорректное состояние вызова.');
    const sinceMs=since?new Date(since).getTime():null,untilMs=until?new Date(until).getTime():null;
    if([sinceMs,untilMs].some(value=>value!==null&&(!Number.isFinite(value)||value<0)))throw new Error('Некорректная дата периода.');
    if(sinceMs!==null&&untilMs!==null&&sinceMs>untilMs)throw new Error('Начало периода должно быть не позже конца.');
    const show=async(offset=0)=>{
      const params=new URLSearchParams({project_id:project,offset:String(offset),limit:'20'});
      if(status)params.set('status',status);if(sinceMs!==null)params.set('since_ms',String(sinceMs));if(untilMs!==null)params.set('until_ms',String(untilMs));
      if(filter)params.set('guardrail',filter);if(hash)params.set('guardrail_policy_sha256',hash);
      const result=await api('observability/traces?'+params);
      if(!current())return;
      if(result.traces.some(trace=>trace.project_id!==project))throw new Error('Получены вызовы другого проекта.');
      const controls=node('div'),previous=node('button','Предыдущие'),next=node('button','Следующие');
      previous.disabled=offset===0;next.disabled=offset+20>=result.total;
      previous.onclick=()=>show(Math.max(0,offset-20)).catch(fail);next.onclick=()=>show(offset+20).catch(fail);controls.append(previous,next);
      panel.replaceChildren(node('h3','Найдено вызовов: '+result.total),...(result.traces.length?result.traces.map(trace=>traceView(trace)):[node('p','Вызовов по выбранному фильтру нет.')]),controls);
    };await show();
  }catch(error){if(current())panel.replaceChildren(node('p',error.message));}
};
$('trace-trash').onclick=async()=>{clearTraceBulk();try{
  const project=$('project').value,panel=$('trace-results');
  const show=async(offset=0)=>{
    const result=await api('observability/traces?removed=true&project_id='+encodeURIComponent(project)+'&offset='+offset+'&limit=20');
    if(project!==$('project').value)return;
    const controls=node('div'),previous=node('button','Предыдущие'),next=node('button','Следующие');
    previous.disabled=offset===0;next.disabled=offset+20>=result.total;
    previous.onclick=()=>show(Math.max(0,offset-20)).catch(fail);next.onclick=()=>show(offset+20).catch(fail);controls.append(previous,next);
    panel.replaceChildren(node('h3','Корзина вызовов · '+result.total),...result.traces.map(trace=>traceView(trace,0,true)),controls);
  };await show();
}catch(error){fail(error);}};
$('judge-runs-refresh').onclick=async()=>{clearTraceBulk();try{
  const project=$('project').value,panel=$('trace-results');
  const show=async(offset=0)=>{
    const result=await api('evaluation/judge-runs?project_id='+encodeURIComponent(project)+'&offset='+offset+'&limit=20');
    if(project!==$('project').value)return;
    const rows=result.runs.map(run=>{
      const row=node('details');row.append(node('summary',run.id+' · '+(evaluationLabels[run.status]||run.status)+' · '+run.completed+'/'+run.sample_count));
      row.append(node('p',run.mean_score==null?'Общая оценка пока недоступна.':'Средняя оценка: '+run.mean_score));
      const open=node('button','Открыть результаты'),trace=node('button','Открыть трассу'),content=node('div');
      open.onclick=async()=>{try{const detail=await api('evaluation/judge-runs/'+encodeURIComponent(run.id));if(project!==$('project').value)return;content.replaceChildren(node('pre',JSON.stringify(detail,null,2)));}catch(error){fail(error);}};
      trace.onclick=async()=>{try{const detail=await api('observability/traces/'+encodeURIComponent(run.trace_id));if(project!==$('project').value)return;if(detail.project_id!==project)throw new Error('Трасса принадлежит другому проекту.');const view=traceView(detail);view.open=true;content.replaceChildren(view);}catch(error){fail(error);}};
      row.append(open,trace);
      if(run.status==='running'){const cancel=node('button','Остановить оценку');cancel.onclick=async()=>{cancel.disabled=true;try{await api('evaluation/judge-runs/'+encodeURIComponent(run.id)+'/cancel',{});await show(offset);}catch(error){cancel.disabled=false;fail(error);}};row.append(cancel);}
      row.append(content);return row;
    });
    const controls=node('div');
    if(offset>0){const back=node('button','Предыдущие оценки');back.onclick=()=>show(Math.max(0,offset-20)).catch(fail);controls.append(back);}
    if(offset+20<result.total){const next=node('button','Следующие оценки');next.onclick=()=>show(offset+20).catch(fail);controls.append(next);}
    panel.replaceChildren(node('p','Сохранённых запусков: '+result.total),...rows,controls);
  };await show();
}catch(error){fail(error);}};
$('trace-refresh').onclick=async()=>{
  clearTraceBulk();
  if(!active)return;
  const sessionId=active;
  try{
    const result=await api('observability/traces?session_id='+encodeURIComponent(sessionId)+'&limit=20');
    if(active!==sessionId)return;
    const rows=[];
    for(const trace of result.traces)rows.push(traceView(trace));
    $('trace-results').replaceChildren(...(rows.length?rows:[node('p','Сохранённых вызовов пока нет.')]));
  }catch(error){fail(error);}
};
let backgroundLauncherSession=null,backgroundStartPending=false;
function renderBackgroundLauncher(s){
  if(backgroundLauncherSession!==s?.id){backgroundLauncherSession=s?.id||null;$('background-command').value='';$('background-name').value='';$('background-follow-up').value='';$('background-timeout').value='3600';$('background-start-result').replaceChildren();}
  $('background-launcher').hidden=!s||s.folder!=='active'||!['auto','goal'].includes(s.settings.mode);
  $('background-start').disabled=backgroundStartPending;
}
$('background-start').onclick=async()=>{
  const sessionId=active;if(!sessionId||!current||current.folder!=='active'||!['auto','goal'].includes(current.settings.mode)||backgroundStartPending)return;
  const name=$('background-name').value;
  const command=$('background-command').value,followUp=$('background-follow-up').value,timeout=Number($('background-timeout').value);
  const size=text=>new TextEncoder().encode(text).length;
  if(!command.trim()||size(command)>16000||!Number.isSafeInteger(timeout)||timeout<1||timeout>86400||size(followUp)>16000){$('background-start-result').replaceChildren(node('p','Укажите команду до 16000 байт и лимит 1–86400 секунд.'));return;}
  backgroundStartPending=true;$('background-start').disabled=true;
  try{
    if(new TextEncoder().encode(name).length>200)throw new Error('Название превышает 200 байт');
    const body={action:'start',command,timeout};if(name.trim())body.name=name;if(followUp.trim())body.follow_up=followUp;
    const result=await api('sessions/'+sessionId+'/background',body);
    if(active===sessionId){$('background-start-result').replaceChildren(node('p','Задача запущена: '+result.id));if($('background-command').value===command&&$('background-follow-up').value===followUp&&$('background-name').value===name){$('background-command').value='';$('background-name').value='';$('background-follow-up').value='';}await poll();}
  }catch(error){if(active===sessionId)$('background-start-result').replaceChildren(node('p',error.message));}
  finally{backgroundStartPending=false;$('background-start').disabled=false;}
};
let lastBackground = '';
let backgroundOutputSession=null;
const backgroundOutputCache=new Map();
const backgroundOutputRequests=new Map();
let backgroundVisibleTaskIds=new Set();
function renderBackground(s) {
  const panel=$('background-tasks'),tasks=s.background?.tasks||[];
  if(backgroundOutputSession!==s.id){backgroundOutputSession=s.id;backgroundOutputCache.clear();backgroundOutputRequests.clear();}
  const taskIds=new Set(tasks.map(task=>task.id));
  backgroundVisibleTaskIds=taskIds;
  for(const id of backgroundOutputCache.keys())if(!taskIds.has(id))backgroundOutputCache.delete(id);
  for(const id of backgroundOutputRequests.keys())if(!taskIds.has(id))backgroundOutputRequests.delete(id);
  const key=JSON.stringify([s.id,tasks]);
  if(key===lastBackground)return;
  lastBackground=key;panel.hidden=!tasks.length;
  const running=tasks.filter(task=>task.status==='running').length;
  const rows=[node('h3','Фоновые задачи · '+tasks.length+' · выполняется '+running)];
  const totals=s.background?.summary;
  if(totals?.duration_semantics==='sum_task_elapsed'&&Number.isSafeInteger(totals.known_duration_ms)&&totals.known_duration_ms>=0&&Number.isSafeInteger(totals.unknown_duration_count)&&totals.unknown_duration_count>=0)rows[0].append(node('small','Сумма известных длительностей: '+(totals.known_duration_ms/1000).toFixed(2)+' с · без длительности: '+totals.unknown_duration_count+'. Длительности параллельных задач складываются.'));
  const labels={running:'Выполняется',completed:'Завершено',failed:'Ошибка',cancelled:'Отменено',timed_out:'Время истекло',interrupted:'Прервано при перезапуске'};
  for(const task of tasks){
    const row=node('div'),summary=node('span',(task.name?task.name+' · ':'')+task.id+' · '+(labels[task.status]||task.status)+(Number.isSafeInteger(task.duration_ms)&&task.duration_ms>=0?' · '+(task.duration_ms/1000).toFixed(2)+' с':'')),output=node('pre');
    if(Number.isSafeInteger(task.exit_code))summary.textContent+=' · код '+task.exit_code;
    else if(Number.isSafeInteger(task.signal)&&task.signal>0)summary.textContent+=' · сигнал '+task.signal;
    else if(task.status==='failed')summary.textContent+=' · код завершения неизвестен';
    if(task.error==='receipt_storage_error')summary.textContent+=' · результат не удалось сохранить';
    if(Number.isSafeInteger(task.started_ms)&&task.started_ms>0){const started=new Date(task.started_ms);if(!Number.isNaN(started.getTime()))summary.textContent+=' · запуск '+started.toLocaleString();}
    if(task.stdout_truncated||task.stderr_truncated)summary.textContent+=' · вывод сокращён';
    output.textContent=backgroundOutputCache.get(task.id)||'';
    const sessionId=s.id;
    const show=node('button',backgroundOutputCache.has(task.id)?'Обновить вывод':'Показать вывод');
    show.onclick=async()=>{const requestToken={};backgroundOutputRequests.set(task.id,requestToken);try{
      const result=await api('sessions/'+sessionId+'/background',{action:'output',task_id:task.id});
      const text=(result.stdout||'')+(result.stderr?'\nОшибки:\n'+result.stderr:'')+(result.stdout_truncated||result.stderr_truncated?'\nВывод сокращён.':'');
      if(backgroundOutputSession!==sessionId||!backgroundVisibleTaskIds.has(task.id)||backgroundOutputRequests.get(task.id)!==requestToken)return;
      backgroundOutputCache.set(task.id,text);
      output.textContent=text;
      lastBackground='';
    }catch(error){if(backgroundOutputSession===sessionId&&backgroundOutputRequests.get(task.id)===requestToken&&backgroundVisibleTaskIds.has(task.id))fail(error);}};
    row.append(summary,show);
    if(task.status==='running'){
      const cancel=node('button','Отменить');cancel.onclick=async()=>{try{
        cancel.disabled=true;await api('sessions/'+sessionId+'/background',{action:'cancel',task_id:task.id});
      }catch(error){cancel.disabled=false;fail(error);}};row.append(cancel);
    }
    row.append(output);
    if(task.status!=='running'){
      const remove=node('button','Убрать запись');remove.onclick=async()=>{if(remove.disabled)return;remove.disabled=true;try{
        await api('sessions/'+sessionId+'/background',{action:'cleanup',task_id:task.id});
        if(backgroundOutputSession===sessionId){backgroundOutputCache.delete(task.id);backgroundOutputRequests.delete(task.id);backgroundVisibleTaskIds.delete(task.id);lastBackground='';}
      }catch(error){remove.disabled=false;fail(error);}};row.append(remove);
    }
    rows.push(row);
  }
  const cleanup=node('button','Убрать завершённые');cleanup.onclick=async()=>{try{
    await api('sessions/'+s.id+'/background',{action:'cleanup'});
  }catch(error){fail(error);}};rows.push(cleanup);
  for(const includeOutputs of [false,true]){
    const download=node('button',includeOutputs?'Экспорт с выводом команд':'Экспорт истории');
    download.onclick=async()=>{if(download.disabled)return;download.disabled=true;try{
      const packet=await api('sessions/'+s.id+'/background',{action:'export',include_outputs:includeOutputs});
      if(backgroundOutputSession!==s.id)return;
      if(packet.kind!=='background_export'||packet.schema_version!==1||packet.session_id!==s.id||packet.outputs_included!==includeOutputs||!Array.isArray(packet.tasks)||packet.tasks.length>128||packet.tasks.some(task=>!task||typeof task!=='object'||['command','follow_up','error',...(!includeOutputs?['stdout','stderr']:[])].some(key=>Object.hasOwn(task,key))))throw new Error('Экспорт не соответствует выбранному чату');
      const url=URL.createObjectURL(new Blob([JSON.stringify(packet,null,2)],{type:'application/json'})),link=node('a');link.href=url;link.download='background-'+s.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }catch(error){if(backgroundOutputSession===s.id)fail(error);}finally{download.disabled=false;}};rows.push(download);
  }
  panel.replaceChildren(...rows);
}
let bookmarkEditing=null;
$('bookmark-cancel').onclick=()=>$('bookmark-edit-dialog').close();
$('bookmark-save').onclick=async()=>{if(!bookmarkEditing)return;const editing=bookmarkEditing;try{
  await api('sessions/'+editing.sessionId+'/bookmarks',{message_index:editing.messageIndex,label:$('bookmark-label').value});
  $('bookmark-edit-dialog').close();bookmarkEditing=null;await poll();await listChats();
}catch(error){fail(error);}};
function renderBookmarks(s){
  const rows=[];
  for(const mark of s.bookmarks||[]){
    const row=node('div'),jump=node('button',mark.label);
    jump.onclick=()=>{const message=$('messages').querySelector('[data-message-index="'+mark.message_index+'"]');if(!message){notify('Сообщение не найдено');return;}for(let parent=message.parentElement;parent&&parent!==$('messages');parent=parent.parentElement){if(parent.tagName==='DETAILS')parent.open=true;}message.scrollIntoView({block:'center',behavior:'smooth'});};
    const remove=node('button','Убрать');remove.onclick=async()=>{try{await del('sessions/'+s.id+'/bookmarks/'+mark.message_index);await poll();await listChats();}catch(error){fail(error);}};
    row.append(jump,remove);rows.push(row);
  }
  $('conversation-bookmarks').replaceChildren(...(rows.length?rows:[node('p','Отметьте важное сообщение кнопкой «Закладка».')]));
}
function render(s) {
  renderBookmarks(s);
  renderContextStatistics(s);
  renderBackgroundLauncher(s);renderBackground(s);
  if(traceSession!==s.id){traceSession=s.id;$('trace-results').replaceChildren();}

  $('compact').disabled=s.status==='running'||s.folder&&s.folder!=='active';$('compact-summary').textContent=s.compaction?.summary||'Контекст ещё не сжат.';
  current=s;$('title').textContent=s.title;renderStatus(s.status);
  $('branch-origin').hidden=!s.parent;
  if(s.parent){const origin=node('button','← Исходный разговор');origin.onclick=()=>openChat(s.parent.session_id).catch(fail);$('branch-origin').replaceChildren(origin,node('span','Независимая ветка · файлы проекта общие'));}
  if(s.error){$('error').textContent=s.error;$('error').hidden=false;}else{$('error').hidden=true;}
  const notice=s.folder&&s.folder!=='active'?'Разговор в архиве или корзине. Откройте «Разговор» → «Восстановить», чтобы продолжить.':s.notice;
  $('notice').textContent=notice||'';$('notice').hidden=!notice;
  const serialized=JSON.stringify([s.messages,s.status])+'|'+pendingRev;
  if(serialized!==lastMessages){
    const pane=$('messages'),atBottom=pane.scrollHeight-pane.scrollTop-pane.clientHeight<120;
    const detailState=new Map([...pane.querySelectorAll('details[data-open-key]')].map(d=>[d.dataset.openKey,d.open]));
    let built=buildMessageNodes(s,detailState);
    // Неподтверждённая отправка живёт одним узлом в конце транскрипта: сервер
    // ещё не добавил сообщение, но пользователь уже должен его видеть.
    if(pendingSend&&!echoedUser(s,pendingSend.text)){
      const ghost=renderSingleMessage(s,{role:'user',content:pendingSend.text,images:pendingSend.images},s.messages.length,()=>'0:0',new Map());
      ghost.classList.add('pending-send');
      // Ветвиться от неподтверждённого сообщения нельзя — кнопки убираем,
      // а не прячем: они не должны попасть ни в разметку, ни в фокус.
      ghost.querySelector('.message-actions')?.remove();
      built.push(ghost);
    }else if(pendingSend){pendingSend=null;pendingRev++;}
    // Swapping only the units that were rebuilt keeps the scroll position and
    // every open `<details>` of the untouched messages.
    if(pane.children.length!==built.length)pane.replaceChildren(...built);
    else built.forEach((n,k)=>{if(pane.children[k]!==n)pane.children[k].replaceWith(n);});
    lastMessages=serialized;if(atBottom)pane.scrollTop=pane.scrollHeight;
  }
  $('queue').replaceChildren(...s.queue.map((p,i)=>node('div',`${i+1}. ${p.text.slice(0,100)}`,'chip')),...s.steering.map(t=>node('div','Steer: '+t.slice(0,100),'chip')));
  if(s.queue.length||s.steering.length){const b=node('button','Очистить очередь');b.onclick=()=>control('clear_queue').catch(fail);$('queue').append(b);}
  renderPlan(s.plan);renderPlanCheckpoints(s);
  $('plan-add').hidden=s.folder!=='active';
  renderUsage(s);
}
let lastPlanCheckpointSignature='';
function renderPlanCheckpoints(session){
  const signature=session?session.id+':'+(session.plan_revision||0)+':'+(session.goal?.id||''):'';
  if(signature===lastPlanCheckpointSignature)return;lastPlanCheckpointSignature=signature;
  $('plan-revision').textContent=session?'План: версия '+(session.plan_revision||0)+'. Критерии и подтверждения заявлены пользователем или агентом.':'';
  const entries=session?.plan_checkpoints||[],rows=[];
  if(entries[0]?.revision>1)rows.push(node('p','Ранние контрольные точки не входят в сохранённое окно истории.'));
  for(const entry of [...entries].reverse()){
    const details=node('details');details.append(node('summary','Версия '+entry.revision+' · '+(entry.source==='user'?'изменение пользователя':entry.source==='agent'?'изменение агента':'восстановление старого плана')));
    if(entry.goal_id)details.append(node('p',entry.goal_id===session.goal?.id?'Текущая цель':'Предыдущая цель'));
    const origin=entry.goal_origin,objective=origin&&origin.id===entry.goal_id&&Number.isSafeInteger(origin.message_index)?session.messages?.[origin.message_index]:null;
    if(objective?.role==='user')details.append(node('p','Исходная задача: '+String(objective.content||'').slice(0,400)));

    for(const step of entry.steps){details.append(node('p',step.title+' · '+step.status));for(const text of step.acceptance||[])details.append(node('p','Критерий: '+text));for(const text of step.evidence||[])details.append(node('p','Заявленное подтверждение: '+text));}rows.push(details);
  }
  $('plan-checkpoints').replaceChildren(...rows);
}
function renderPlan(items,force){
  const sig=JSON.stringify({items,mode:current?.settings.mode,revision:current?.plan_revision});
  if(!force&&sig===lastPlan)return;
  lastPlan=sig;
  const list=$('plan');
  list.replaceChildren();
  if(!items.length){list.append(node('li','План появится во время работы','muted'));return;}
  items.forEach((p,i)=>{
    const li=node('li',undefined,p.status);
    const status=node('button',p.status==='completed'?'✓':p.status==='in_progress'?'◐':'○','plan-status');
    status.title='Изменить статус';
    status.onclick=()=>{const reopen=p.status==='completed';p.status=p.status==='pending'?'in_progress':p.status==='in_progress'?'completed':'pending';savePlan(items,reopen);};
    const title=node('span',p.title,'plan-title');
    title.title='Кликните, чтобы изменить';
    title.onclick=()=>{
      if(li.querySelector('input'))return;
      const input=node('input');input.value=p.title;input.maxLength=300;
      let settled=false;
      const done=save=>{if(settled)return;settled=true;if(save){const v=input.value.trim();if(v){p.title=v;savePlan(items);return;}}renderPlan(items,true);};
      input.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();done(true);}else if(e.key==='Escape'){e.preventDefault();done(false);}};
      input.onblur=()=>done(true);
      li.replaceChild(input,title);input.focus();input.select();
    };
    const del=node('button','×','plan-delete');del.title='Удалить шаг';
    del.onclick=()=>{const reopen=p.status==='completed';items.splice(i,1);savePlan(items,reopen);};
    li.append(status,title,del);
    if(current?.settings.mode==='goal'||p.acceptance?.length||p.evidence?.length){
      const details=node('details'),summary=node('summary','Критерии и заявленные подтверждения'),criteria=node('textarea'),evidence=node('textarea');
      criteria.setAttribute('aria-label','Критерии этапа');evidence.setAttribute('aria-label','Подтверждения этапа');criteria.value=(p.acceptance||[]).join('\n');evidence.value=(p.evidence||[]).join('\n');
      criteria.disabled=evidence.disabled=current?.settings.mode==='goal'&&p.status==='completed';
      criteria.onchange=()=>{p.acceptance=criteria.value.split('\n').map(value=>value.trim()).filter(Boolean);savePlan(items);};
      evidence.onchange=()=>{p.evidence=evidence.value.split('\n').map(value=>value.trim()).filter(Boolean);savePlan(items);};
      details.append(summary,node('p','До четырёх строк в каждом поле. Это заявленные основания, а не независимое подтверждение выполнения.'),criteria,evidence);li.append(details);
    }
    list.append(li);
  });
}
async function savePlan(items,allowReopen=false){
  if(!active)return;
  try{await api(`sessions/${active}/plan`,{steps:items,base_revision:current?.plan_revision||0,allow_reopen:allowReopen});}
  catch(e){notify(e.message);}
  lastPlan='';
  poll().catch(()=>{});
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
function applyCommand(text){
  const m=text.match(/^\/(\w+)\s*([\s\S]*)$/);
  if(!m)return false;
  const cmd=m[1].toLowerCase(),arg=m[2].trim();
  if(['chat','plan','auto','goal','swarm'].includes(cmd)){
    $('mode').value=cmd;modeHelp();savePrefs();
    if(arg){$('prompt').value=arg;return false;}
    notify('Режим: '+cmd);return true;
  }
  switch(cmd){
    case 'new': resetChat(); return true;
    case 'compact': control('compact').catch(fail); return true;
    case 'export': if(current){$('export').onclick();}else notify('Нет открытого разговора'); return true;
    case 'help': notify('/chat /plan /auto /goal /swarm /new /compact /export'); return true;
  }
  return false;
}
async function control(kind, actionText) {
  $('error').hidden=true;
  if(current&&current.folder&&current.folder!=='active'&&['send','send_now','steer','resume','compact','retry_member'].includes(kind))throw new Error('Разговор в архиве или корзине. Восстановите его, чтобы продолжить.');
  const sends=['send','send_now','steer'].includes(kind);
  let text=$('prompt').value.trim();
  if(sends&&text.startsWith('/')){
    if(applyCommand(text)){$('prompt').value='';return;}
    text=$('prompt').value.trim();
  }
  if(sends&&layoutFix&&!text.startsWith('/')){autoFixField(false);text=$('prompt').value.trim();}
  const snapshot=settings();
  if(sends){
    if(draftTexts.length)text+='\n\n'+draftTexts.map(f=>`<attached-file name=${JSON.stringify(f.name)}>\n${f.text}\n</attached-file>`).join('\n\n');
    if(!text && draftImages.length)text='Посмотри на приложенные изображения.';
    if(!text)throw new Error('Введите сообщение или приложите файл.');
    if(!snapshot.model)throw new Error('Выберите или введите Model ID.');
    if(kind==='steer'&&draftImages.length)throw new Error('Для изображений используйте Send now или очередь.');
  }
  if(!active){if(!sends)return;active=(await api('sessions',snapshot)).id;savePrefs();}
  // Сообщение показываем сразу, не дожидаясь ответа сервера: отклик на клик
  // важнее подтверждения. Откат — тем же путём, если действие отклонено.
  const optimistic=kind==='send'||kind==='send_now';
  if(optimistic){pendingSend={text,images:draftImages.slice()};pendingRev++;poll().catch(()=>{});}
  try{
    await api(`sessions/${active}/actions`,{kind,text:sends?text:(actionText||''),settings:(sends&&kind!=='steer')||['resume','compact'].includes(kind)?snapshot:undefined,images:sends?draftImages:[]});
  }catch(e){
    if(optimistic){pendingSend=null;pendingRev++;poll().catch(()=>{});}
    throw e;
  }
  if(sends){$('prompt').value='';draftImages=[];draftTexts=[];renderAttachments();}
  await poll();await listChats();
}
function modeHelp() {
  const mode=$('mode').value;$('writes-label').hidden=!['auto','goal'].includes(mode);
  $('swarm-config').hidden=mode!=='swarm';
  $('mode-help').textContent={chat:'Chat · чтение контекста и ответы',plan:'Plan · исследование и план, без записи',auto:'Auto · инструменты до результата или лимита',goal:'Goal · автономная работа к цели с планом и проверкой',swarm:'Swarm · волна независимых агентов и сводный MASTER, без записи'}[mode]||'';
  if(mode==='swarm'){if(!swarmPanel().children.length)fillSwarmDefaults(false);fillSynthProviders();updateSwarmCost();}
}
const LAYOUT_EN='qwertyuiopasdfghjklzxcvbnm';
const LAYOUT_RU='йцукенгшщзфывапролдячсмить';
const LAYOUT_EN_FULL="qwertyuiop[]asdfghjkl;'zxcvbnm,./`";
const LAYOUT_RU_FULL='йцукенгшщзхъфывапролджэячсмитьбю.ё';
const EN_VOWELS='aeiouy',RU_VOWELS='аеёиоуыэюя';
const layoutSkip=new Set(['src','tmp','cfg','http','https','html','css','json','sql','pdf','xml','url','api','id','ui','db','fs','cli','env','var','fn','mod','pub','mut','ref','git','npm','cargo','js','ts','rs','py','md','png','jpg','gif','svg','csv','txt','str','buf','req','res','ctx','ptr','len','idx','std','os','io','dbg']);
function layoutMap(text,from,to,strict){
  let out='';
  for(const ch of text){
    const i=from.indexOf(ch.toLowerCase());
    if(i<0){if(strict)return null;out+=ch;continue;}
    const mapped=to[i];
    out+=ch!==ch.toLowerCase()?mapped.toUpperCase():mapped;
  }
  return out;
}
function layoutLooksWord(s,vowels){
  const letters=[...s.toLowerCase()].filter(c=>/[a-zа-яё]/.test(c));
  if(letters.length<3)return false;
  let count=0,run=0,max=0;
  for(const c of letters){
    if(vowels.includes(c)){count++;run=0;}else{run++;if(run>max)max=run;}
  }
  const ratio=count/letters.length;
  return count>0&&ratio>=0.18&&ratio<=0.65&&max<=4;
}
function layoutFixToken(word){
  if(word.length<3)return null;
  if(layoutSkip.has(word.toLowerCase()))return null;
  if(/[a-z][A-Z]/.test(word))return null;
  if(/^[a-z]+$/i.test(word)){
    const ru=layoutMap(word,LAYOUT_EN,LAYOUT_RU,true);
    return ru&&!layoutLooksWord(word,EN_VOWELS)&&layoutLooksWord(ru,RU_VOWELS)?ru:null;
  }
  if(/^[а-яё]+$/i.test(word)){
    const en=layoutMap(word,LAYOUT_RU,LAYOUT_EN,true);
    return en&&!layoutLooksWord(word,RU_VOWELS)&&layoutLooksWord(en,EN_VOWELS)?en:null;
  }
  return null;
}
function layoutFixLastWord(){
  const ta=$('prompt'),caret=ta.selectionStart,value=ta.value;
  if(caret<2||!/^\s$/.test(value[caret-1]))return;
  let start=caret-1;
  while(start>0&&/[A-Za-zА-Яа-яЁё]/.test(value[start-1]))start--;
  const word=value.slice(start,caret-1);
  const fixed=layoutFixToken(word);
  if(!fixed||fixed===word)return;
  ta.value=value.slice(0,start)+fixed+value.slice(caret-1);
  const pos=caret+(fixed.length-word.length);
  ta.setSelectionRange(pos,pos);
}
function convertField(){
  const ta=$('prompt'),original=ta.value;
  if(!original.trim())return;
  const cyr=(original.match(/[а-яё]/gi)||[]).length;
  const lat=(original.match(/[a-z]/gi)||[]).length;
  const toRu=lat>=cyr;
  const fixed=layoutMap(original,toRu?LAYOUT_EN_FULL:LAYOUT_RU_FULL,toRu?LAYOUT_RU_FULL:LAYOUT_EN_FULL,false);
  if(!fixed||fixed===original){notify('Раскладка не изменена');return;}
  ta.value=fixed;
  ta.focus();
  showUndo('Раскладка исправлена',()=>{ta.value=original;});
}
function layoutConvertTokens(text,completeOnly){
  const re=/[A-Za-zА-Яа-яЁё]+/g;
  let out='',last=0,changed=false,m;
  while((m=re.exec(text))){
    out+=text.slice(last,m.index);
    const word=m[0],atEnd=m.index+word.length===text.length;
    const fixed=completeOnly&&atEnd?null:layoutFixToken(word);
    if(fixed&&fixed!==word){out+=fixed;changed=true;}else out+=word;
    last=m.index+word.length;
  }
  out+=text.slice(last);
  return {text:out,changed};
}
function autoFixField(completeOnly){
  const ta=$('prompt'),original=ta.value;
  if(!original)return;
  const caret=ta.selectionStart??original.length;
  const full=layoutConvertTokens(original,completeOnly);
  if(!full.changed||full.text===original)return;
  const prefix=layoutConvertTokens(original.slice(0,caret),completeOnly);
  ta.value=full.text;
  const pos=Math.max(0,Math.min(full.text.length,caret+(prefix.text.length-original.slice(0,caret).length)));
  ta.setSelectionRange(pos,pos);
  showUndo('Раскладка исправлена',()=>{ta.value=original;});
}
let layoutTimer=null;
function scheduleLayoutFix(){
  if(!layoutFix)return;
  clearTimeout(layoutTimer);
  layoutTimer=setTimeout(()=>autoFixField(true),600);
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
  const id=$('model').value.trim();
  const model=(catalogs.get($('provider').value)||[]).find(m=>m.id===id);
  $('model-info').hidden=!model&&!id;
  if(!model){
    $('model-info').textContent=id?`Модель чата: ${selectedModelLabel()}. Сведения о контексте, цене и инструментах появятся после «Обновить список».`:'';
    return;
  }
  const parts=[`Модель чата: ${model.name||model.id}`];
  if(model.context_length)parts.push(`Контекст: ${Number(model.context_length).toLocaleString('ru')}`);
  if(model.top_provider?.max_completion_tokens)parts.push(`Ответ до ${model.top_provider.max_completion_tokens}`);
  if(model.architecture?.input_modalities)parts.push('Вход: '+model.architecture.input_modalities.join(', '));
  if(model.supported_parameters)parts.push(model.supported_parameters.includes('tools')?'Инструменты поддерживаются':'Инструменты не указаны в каталоге');
  for(const [key,label] of [['prompt','вход'],['completion','выход']]) {
    const raw=model.pricing?.[key];if(raw!==undefined&&raw!==null&&raw!==''&&Number.isFinite(Number(raw)))parts.push(`$${(Number(raw)*1000000).toLocaleString('en',{maximumFractionDigits:4})} / 1M ${label}`);
  }
  $('model-info').textContent=parts.join(' · ');
}
async function refreshModels(provider, {selectFirst=false}={}) {
  const selected = provider || $('provider').value;
  const btn = $('load-models');
  btn.disabled = true; btn.textContent = 'Загрузка…';
  try {
    const v = await api(`providers/${selected}/models`);
    if ($('provider').value !== selected) return null;
    catalogs.set(selected, v.catalog || []);
    $('models').replaceChildren(...v.models.map(m=>{const o=node('option');o.value=m;return o;}));
    renderModelInfo();renderModelList();
    if (selectFirst && v.models.length && !$('model').value.trim()) {
      $('model').value = v.models[0];
      renderModelInfo();refreshContextModel();
      if (typeof savePrefs === 'function') savePrefs();
    }
    return v.models;
  } finally {
    btn.disabled = false; btn.textContent = 'Обновить список';
  }
}
$('model').oninput=()=>{renderModelInfo();refreshContextModel();savePrefs();};
async function renderPlugins(){
  const data=await api('plugins');
  const items=data.plugins||[];
  $('plugin-list').replaceChildren(...items.map(p=>{
    const row=node('div',undefined,'plugin-row');
    const kindBadge=p.kind==='skill'?'skill':'mcp';
    const status=p.kind==='skill'?(p.enabled?'активен':'выключен'):(p.connected?`● ${p.tools} инструментов`:p.enabled?'подключение…':'выключен');
    if(p.error)row.append(node('p',p.error,'muted'));
    row.append(node('span',`${p.name} · [${kindBadge}] · ${status}`,'muted'));
    const actions=node('div',undefined,'key-actions');
    if(p.kind!=='skill'){const reload=node('button','Переподключить');reload.onclick=async()=>{reload.disabled=true;try{await api(`plugins/${p.id}/reload`,{});await renderPlugins();}catch(e){notify(e.message);reload.disabled=false;}};actions.append(reload);}
    const del=node('button','Удалить');del.onclick=async()=>{del.disabled=true;try{await api(`plugins/${p.id}/delete`,{});await renderPlugins();}catch(e){notify(e.message);del.disabled=false;}};actions.append(del);
    row.append(actions);
    return row;
  }));
  if(!items.length)$('plugin-list').replaceChildren(node('div','Нет плагинов','muted'));
}
async function showConnections(){
  await loadConfig();
  doctorEpoch++;$('doctor-result').replaceChildren();
  $('doctor-provider').replaceChildren(...config.providers.map(provider=>{const option=node('option',provider.name);option.value=provider.id;return option;}));
  $('doctor-provider').value=$('provider').value;$('doctor-model').value=$('model').value.trim();
  renderPlugins().catch(e=>notify(e.message));
  $('key-fields').replaceChildren(...config.providers.filter(p=>p.id!=='local').map(p=>{
    const div=node('div',undefined,'key-row'),label=node('label',p.name+' · '+(p.key_env||'API key')),input=node('input');
    input.type='password';input.autocomplete='new-password';input.dataset.provider=p.id;input.placeholder=p.configured?'Введите новый ключ для замены':'API key';label.append(input);
    const sources={environment:'Из переменной окружения',system:'Из системного хранилища',memory:'Только в памяти',none:'Ключ не задан'};
    div.append(label,node('small',(sources[p.key_source]||'')+(p.saved?' · сохранён для перезапуска':'')));
    if(p.key_error)div.append(node('p',p.key_error,'muted'));
    const storage=node('label'),persist=node('input');persist.type='checkbox';persist.dataset.persist=p.id;persist.checked=p.saved;persist.disabled=!config.credential_storage;
    storage.append(persist,node('span',config.credential_storage?'Системное хранилище (без отметки — только память, сохранённый ключ удаляется)':'Системное хранилище недоступно; используйте ENV'));div.append(storage);
    const actions=node('div',undefined,'key-actions');
    if(config.credential_storage&&p.configured){const save=node('button','Сохранить ключ в системном хранилище');save.onclick=async()=>{save.disabled=true;try{const replacement=input.value.trim();await api(`providers/${p.id}/key`,replacement?{key:replacement,persist:true}:{key:'',persist:true,use_current:true});await showConnections();notify('Ключ сохранён в системном хранилище');}catch(e){notify(e.message);save.disabled=false;}};actions.append(save);}
    if(p.configured||p.saved){const remove=node('button','Удалить ключ');remove.onclick=async()=>{remove.disabled=true;try{await api(`providers/${p.id}/key`,{key:'',persist:false});await showConnections();notify('Ключ удалён из Studio и системного хранилища. ENV не изменены.');}catch(e){notify(e.message);remove.disabled=false;}};actions.append(remove);}
    if(p.custom){const del=node('button','Удалить провайдера');del.onclick=async()=>{del.disabled=true;try{await api(`providers/${p.id}/delete`,{});await showConnections();await loadConfig();notify('Провайдер удалён');}catch(e){notify(e.message);del.disabled=false;}};actions.append(del);}
    div.append(actions);return div;
  }));
  if(!$('connection-dialog').open)$('connection-dialog').showModal();
}
let doctorEpoch=0;
const doctorMessages={
  ready:['Каталог моделей доступен.','Если указана модель, она найдена в каталоге. Это не проверка генерации ответа.'],
  model_missing:['Модель не найдена в каталоге.','Проверьте её название или обновите список моделей.'],
  missing_key:['Ключ провайдера не задан.','Введите и сохраните ключ в окне подключений.'],
  credential_store_error:['Не удалось получить ключ из хранилища.','Проверьте доступ к системному хранилищу или сохраните ключ заново.'],
  invalid_endpoint:['Адрес подключения некорректен.','Проверьте адрес сервера. Логин, пароль и параметры в адресе не поддерживаются.'],
  unreachable:['Сервер недоступен.','Проверьте адрес, сеть и запуск локального сервера.'],
  timeout:['Сервер не ответил за 10 секунд.','Проверьте сеть и нагрузку сервера; затем повторите проверку.'],
  unauthorized:['Сервер отклонил доступ.','Проверьте ключ и его разрешения.'],
  rate_limited:['Сервер ограничил частоту запросов.','Повторите проверку позднее.'],
  redirect_rejected:['Сервер перенаправляет запрос.','Укажите конечный адрес подключения.'],
  catalog_unavailable:['Список моделей по этому адресу недоступен.','Проверьте адрес API; некоторые серверы не предоставляют список моделей.'],
  provider_error:['Сервер вернул ошибку.','Проверьте состояние сервера или повторите позднее.'],
  invalid_catalog:['Сервер вернул некорректный список моделей.','Проверьте совместимость сервера со списком моделей API.'],
  catalog_too_large:['Список моделей превышает допустимый размер.','Сервер должен вернуть не более 1 MiB и 10 000 моделей.'],
  empty_catalog:['Сервер вернул пустой список моделей.','Убедитесь, что на сервере загружена модель.']
};
function clearDoctor(){doctorEpoch++;$('doctor-result').replaceChildren();}
$('doctor-provider').onchange=()=>{clearDoctor();$('doctor-model').value=$('doctor-provider').value===$('provider').value?$('model').value.trim():'';};
$('doctor-model').oninput=clearDoctor;
$('doctor-check').onclick=async()=>{
  const button=$('doctor-check'),provider=$('doctor-provider').value,model=$('doctor-model').value.trim(),epoch=++doctorEpoch;
  button.disabled=true;$('doctor-result').replaceChildren(node('p','Проверяем подключение…'));
  try{const result=await api('providers/'+encodeURIComponent(provider)+'/doctor',{model:model||null});
    if(epoch!==doctorEpoch||!$('connection-dialog').open)return;
    if(result.provider_id!==provider||result.generation_calls!==0||result.inference_verified!==false)throw new Error('Некорректный результат диагностики.');
    const message=doctorMessages[result.status];if(!message)throw new Error('Неизвестный результат диагностики.');
    const rows=[node('p',message[0]),node('p',message[1]),node('p','Время проверки: '+result.elapsed_ms+' мс')];
    if(result.model_count!=null)rows.push(node('p','Моделей в каталоге: '+result.model_count));
    if(result.http_status!=null)rows.push(node('p','Ответ сервера: '+result.http_status));
    $('doctor-result').replaceChildren(...rows);
  }catch(error){if(epoch===doctorEpoch)$('doctor-result').replaceChildren(node('p',error.message));}finally{button.disabled=false;}
};
function rootRow(root={alias:'',path:'',writable:false}){
  const div=node('div'),alias=node('input'),path=node('input'),remove=node('button','×'),label=node('label','Запись'),write=node('input');
  alias.placeholder='Имя контекста';alias.value=root.alias;alias.dataset.field='alias';path.placeholder='/абсолютный/путь';path.value=root.path;path.dataset.field='path';write.type='checkbox';write.checked=root.writable;write.dataset.field='writable';label.prepend(write);remove.onclick=()=>div.remove();div.append(alias,path,remove,label);$('root-editor').append(div);
}
function editProject(fresh=false){const p=fresh?{id:'',name:'',instructions:'',roots:[]}:config.projects.find(p=>p.id===$('project').value);editingProject=p.id;$('project-title').value=p.name;$('project-instructions').value=p.instructions;$('root-editor').replaceChildren();p.roots.forEach(rootRow);if(!p.roots.length)rootRow();$('project-dialog').showModal();}
$('context-toggle').onclick=()=>document.querySelector('.context-panel').classList.toggle('open');
$('menu-toggle').onclick=()=>document.querySelector('.sidebar').classList.toggle('open');
$('save-project').onclick=async()=>{try{const roots=[...$('root-editor').children].map(d=>({alias:d.querySelector('[data-field=alias]').value.trim(),path:d.querySelector('[data-field=path]').value.trim(),writable:d.querySelector('[data-field=writable]').checked}));const p=await api('projects',{id:editingProject,name:$('project-title').value.trim(),instructions:$('project-instructions').value,roots});await loadConfig();$('project').value=p.id;renderContext();$('project-dialog').close();if(current&&current.settings.project_id!==p.id)resetChat();else await listChats();notify('Контекст проекта сохранён');}catch(e){notify(e.message);}};
$('save-keys').onclick=async()=>{const button=$('save-keys');button.disabled=true;try{for(const input of $('key-fields').querySelectorAll('input[data-provider]'))if(input.value.trim()){const persist=[...$('key-fields').querySelectorAll('input[data-persist]')].find(p=>p.dataset.persist===input.dataset.provider).checked;await api(`providers/${input.dataset.provider}/key`,{key:input.value.trim(),persist});input.value='';}$('key-fields').replaceChildren();$('connection-dialog').close();await loadConfig();notify('Ключи сохранены с выбранными настройками');}catch(e){notify(e.message);}finally{button.disabled=false;}};
$('load-models').onclick=async()=>{try{const models=await refreshModels($('provider').value);if(models)notify(`${models.length} моделей. Выберите Model ID.`);}catch(e){fail(e);}};
$('provider').onchange=()=>{$('model').value='';$('models').replaceChildren();renderModelInfo();refreshContextModel();savePrefs();refreshModels($('provider').value,{selectFirst:true}).catch(()=>{});};
$('project').onchange=()=>{clearTraceBulk();renderContext();resetChat();savePrefs();};
$('mode').onchange=()=>{modeHelp();refreshContextModel();savePrefs();};
$('steps').oninput=savePrefs;
$('output-tokens').oninput=savePrefs;
$('auto-compact').onchange=savePrefs;
$('compact-threshold').oninput=savePrefs;
$('writes').onchange=savePrefs;
$('json-mode').onchange=savePrefs;
$('swarm-add-member').onclick=()=>{const provider=$('provider').value||((providerList()[0]||{}).id||'');swarmPanel().append(swarmRow({label:'',role:'',provider,model:$('model').value.trim()}));updateSwarmCost();savePrefs();};
$('swarm-fill-defaults').onclick=()=>fillSwarmDefaults(true);
$('swarm-rounds').onchange=$('swarm-critic').onchange=$('swarm-steps').oninput=$('swarm-report-kb').oninput=()=>{updateSwarmCost();savePrefs();};
$('swarm-synth-provider').onchange=savePrefs;
$('swarm-synth-model').oninput=savePrefs;
$('new-chat').onclick=resetChat;$('search').oninput=()=>listChats().catch(fail);$('history-folder').onchange=()=>listChats().catch(fail);$('history-sort').onchange=()=>listChats().catch(fail);$('history-bookmarked').onchange=()=>listChats().catch(fail);
$('compact').onclick=()=>control('compact').catch(fail);
$('send').onclick=()=>control('send').catch(fail);$('send-now').onclick=()=>control('send_now').catch(fail);$('steer').onclick=()=>control('steer').catch(fail);$('stop').onclick=()=>control('stop').catch(fail);$('resume').onclick=()=>control('resume').catch(fail);
$('prompt').onkeydown=e=>{
  if(!$('command-menu').hidden){
    if(e.key==='ArrowDown'){e.preventDefault();commandIndex=(commandIndex+1)%commandMatches.length;highlightCommand();return;}
    if(e.key==='ArrowUp'){e.preventDefault();commandIndex=(commandIndex-1+commandMatches.length)%commandMatches.length;highlightCommand();return;}
    if(e.key==='Enter'&&!(e.metaKey||e.ctrlKey)){e.preventDefault();if(commandMatches[commandIndex])selectCommand(commandMatches[commandIndex].cmd);return;}
    if(e.key==='Tab'){e.preventDefault();if(commandMatches[commandIndex])selectCommand(commandMatches[commandIndex].cmd);return;}
    if(e.key==='Escape'){$('command-menu').hidden=true;return;}
  }
  if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)){e.preventDefault();control('send').catch(fail);}
};
$('prompt').oninput=e=>{
  renderCommandMenu();
  if(!layoutFix||e.isComposing)return;
  if(e.data&&/^\s$/.test(e.data))layoutFixLastWord();
  scheduleLayoutFix();
};
$('prompt').addEventListener('focus',()=>renderCommandMenu());
function toggleLayout(){
  layoutFix=!layoutFix;
  $('layout-toggle').classList.toggle('active',layoutFix);
  savePrefs();
  if(layoutFix){autoFixField(false);notify('Авто-раскладка включена');}
  else notify('Авто-раскладка выключена');
}
document.addEventListener('click',e=>{
  if(e.target.closest('#layout-toggle'))toggleLayout();
  else if(e.target.closest('#layout-convert'))convertField();
});
$('files').onchange=async e=>{try{await attachFiles(e.target.files);}catch(e){fail(e);}e.target.value='';};
$('prompt').addEventListener('paste',e=>{const files=[...e.clipboardData.files];if(files.length){e.preventDefault();attachFiles(files).catch(fail);}});
$('messages').ondragover=e=>e.preventDefault();$('messages').ondrop=e=>{e.preventDefault();attachFiles(e.dataTransfer.files).catch(fail);};
$('connections').onclick=()=>showConnections().catch(fail);$('new-project').onclick=()=>editProject(true);$('edit-project').onclick=()=>editProject();$('add-project').onclick=()=>editProject(true);$('manage-context').onclick=()=>editProject();$('add-root').onclick=()=>rootRow();
$('add-custom-provider').onclick=async()=>{try{const name=$('custom-provider-name').value.trim();const base=$('custom-provider-base').value.trim();if(!name||!base)throw new Error('Укажите название и Base URL.');await api('providers',{name,base});$('custom-provider-name').value='';$('custom-provider-base').value='';await showConnections();notify('Провайдер добавлен');}catch(e){notify(e.message);}};
$('add-plugin').onclick=async()=>{try{const kind=$('plugin-kind').value;const name=$('plugin-name').value.trim();const url=$('plugin-url').value.trim();const instructions=$('plugin-instructions').value.trim();const unlocks_tool=$('plugin-unlocks').value.trim()||undefined;if(!name)throw new Error('Укажите название.');if(kind==='mcp'&&!url)throw new Error('Укажите MCP URL.');if(kind==='skill'&&!instructions)throw new Error('Укажите инструкции.');await api('plugins',{name,kind,url,instructions,unlocks_tool});$('plugin-name').value='';$('plugin-url').value='';$('plugin-instructions').value='';$('plugin-unlocks').value='';await renderPlugins();notify('Плагин добавлен');}catch(e){notify(e.message);}};
$('plugin-kind').onchange=()=>{const kind=$('plugin-kind').value;$('plugin-url-label').hidden=kind==='skill';$('plugin-instructions-label').hidden=kind!=='skill';$('plugin-unlocks-label').hidden=kind!=='skill';};
$('plan-add').onclick=()=>{
  if(!current)return;
  if(current.settings.mode!=='goal'){const items=[...(current.plan||[])];items.push({title:'Новый шаг',status:'pending'});savePlan(items);return;}
  if($('plan').querySelector('[data-goal-draft]'))return;
  const origin=active,revision=current.plan_revision||0,steps=JSON.parse(JSON.stringify(current.plan||[])),draft=node('li'),title=node('input'),criteria=node('textarea'),submit=node('button','Добавить этап'),cancel=node('button','Отмена'),message=node('p');
  draft.setAttribute('data-goal-draft','true');title.value='Новый этап';title.setAttribute('aria-label','Название нового этапа');criteria.setAttribute('aria-label','Критерии нового этапа');
  submit.onclick=async()=>{if(active!==origin||submit.disabled)return;submit.disabled=true;
    try{const acceptance=criteria.value.split('\n').map(value=>value.trim()).filter(Boolean);if(!acceptance.length)throw new Error('Укажите хотя бы один критерий приёмки');
      await api('sessions/'+origin+'/plan',{steps:[...steps,{title:title.value,status:'pending',acceptance,evidence:[]}],base_revision:revision,allow_reopen:false});
      if(active===origin){lastPlan='';await poll();}
    }catch(error){if(active===origin)message.textContent=error.message;}finally{submit.disabled=false;}
  };cancel.onclick=()=>draft.remove();draft.append(title,node('p','До четырёх критериев, по одному на строку.'),criteria,submit,cancel,message);$('plan').append(draft);
};
$('manage-chat').onclick=()=>{if(!current)return;openHistoryDialog(current);};
function openHistoryDialog(r){
  historyTarget=r.id;
  $('chat-title').value=r.title;
  $('archive-chat').disabled=r.status==='running'||r.folder==='archived';
  $('trash-chat').disabled=r.status==='running'||r.folder==='trash';
  $('restore-chat').disabled=!r.folder||r.folder==='active';
  $('delete-chat').hidden=r.folder!=='trash';
  $('history-dialog').showModal();
}
async function updateHistory(kind,text,id){
  const sid=id||active;if(!sid)return;
  if(kind==='move'&&text==='trash'&&!confirm('Переместить разговор в корзину?'))return;
  const from=(kind==='move'&&sid===active)?(current?.folder||'active'):null;
  await api(`sessions/${sid}/actions`,{kind,text});
  $('history-dialog').close();
  if(kind==='move'&&sid===active)$('history-folder').value=text;
  if(sid===active)await poll();
  await listChats();
  refreshFolderCounts().catch(()=>{});
  if(kind==='move'&&from!==null&&from!==text){
    const label=text==='trash'?'Перемещено в корзину':text==='archived'?'Перемещено в архив':'Восстановлено';
    showUndo(label,()=>updateHistory('move',from,sid));
  }
}
async function deleteChat(id){
  if(!confirm('Удалить разговор навсегда? Это действие необратимо.'))return;
  await del('sessions/'+id);
  $('history-dialog').close();
  if(id===active)resetChat();else await listChats();
  refreshFolderCounts().catch(()=>{});
}
$('rename-chat').onclick=()=>updateHistory('rename',$('chat-title').value.trim(),historyTarget).catch(e=>notify(e.message));
$('archive-chat').onclick=()=>updateHistory('move','archived',historyTarget).catch(e=>notify(e.message));
$('trash-chat').onclick=()=>updateHistory('move','trash',historyTarget).catch(e=>notify(e.message));
$('restore-chat').onclick=()=>updateHistory('move','active',historyTarget).catch(e=>notify(e.message));
$('delete-chat').onclick=()=>deleteChat(historyTarget).catch(e=>notify(e.message));
$('import-chat').onclick=()=>$('import-file').click();
$('import-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>9000000)throw new Error('Файл импорта должен быть меньше 9 MB.');let session;try{session=JSON.parse(await file.text());}catch{throw new Error(`Файл «${file.name}» не является корректным JSON.`);}if(!session||typeof session!=='object'||Array.isArray(session))throw new Error(`Файл «${file.name}» должен содержать объект разговора allpaka.`);const imported=await api('sessions/import',{session,project_id:$('project').value});$('search').value='';await openChat(imported.id);notify('Разговор импортирован отдельной копией');refreshFolderCounts().catch(()=>{});}catch(e){notify('Импорт не выполнен');fail(e);}finally{$('import-file').value='';}};
$('export').onclick=()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=`allpaka-${current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
(async()=>{await loadConfig();await listChats();refreshFolderCounts().catch(()=>{});bindSuggestions();modeHelp();refreshContextModel();refreshModels($('provider').value).catch(()=>{});setInterval(()=>poll().catch(fail),650);setInterval(()=>listChats('auto').catch(fail),5000);})().catch(fail);

const presentationKey='allpaka.studio.presentation.v1';
try { const saved=JSON.parse(localStorage.getItem(presentationKey)||'{}');
  if(['brief','normal','detailed','maximum'].includes(saved.verbosity))$('verbosity').value=saved.verbosity;
  if(['hidden','collapsed','expanded'].includes(saved.analysis))$('analysis-display').value=saved.analysis;
} catch {}
function savePresentation(){try{localStorage.setItem(presentationKey,JSON.stringify({verbosity:$('verbosity').value,analysis:$('analysis-display').value}));}catch{}}
$('verbosity').onchange=savePresentation;
$('analysis-display').onchange=()=>{savePresentation();document.querySelectorAll('details.reasoning').forEach(d=>{d.open=$('analysis-display').value==='expanded';});lastMessages='';messageCache=[];if(current)render(current);};

let pluginStatusPolling=false;
setInterval(async()=>{
  if(!$('connection-dialog').open || pluginStatusPolling)return;
  pluginStatusPolling=true;
  try{await renderPlugins();}catch{}finally{pluginStatusPolling=false;}
},1500);

/* ===== Редизайн: рейл, сегменты режима, чип модели, поповеры ===== */
(function redesignShell(){
  const modeSel=$('mode'),seg=$('mode-seg');
  [['chat','Chat'],['plan','Plan'],['auto','Auto'],['goal','Goal'],['swarm','Swarm']].forEach(([value,label])=>{
    const b=node('button',label);b.type='button';b.dataset.mode=value;
    b.onclick=()=>{modeSel.value=value;modeSel.dispatchEvent(new Event('change'));syncShell();};
    seg.append(b);
  });
  function syncShell(){
    seg.querySelectorAll('button').forEach(b=>{const on=b.dataset.mode===modeSel.value;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
    const model=$('model').value.trim();
    $('model-chip').textContent=(model?`${providerName($('provider').value)} · ${model}`:'Выбрать модель')+' ⌄';
  }
  setInterval(syncShell,600);syncShell();
  $('model').addEventListener('input',syncShell);$('provider').addEventListener('change',syncShell);
  const PRESETS={fast:['brief',4096,6],normal:['normal',8192,12],deep:['detailed',16384,24]};
  $('presets').querySelectorAll('button').forEach(b=>b.onclick=()=>{
    const [v,t,s]=PRESETS[b.dataset.preset];
    $('verbosity').value=v;$('output-tokens').value=t;$('steps').value=s;
    for(const id of ['verbosity','output-tokens','steps'])$(id).dispatchEvent(new Event('change'));
    savePrefs();notify('Пресет: '+b.textContent);
  });
  $('rail-new').onclick=()=>$('new-chat').click();
  $('rail-connections').onclick=()=>$('connections').click();
  $('rail-history').onclick=()=>document.body.classList.toggle('nav-collapsed');
  const pops=()=>document.querySelectorAll('details.pop[open],details.menu[open]');
  document.addEventListener('click',e=>pops().forEach(d=>{if(!d.contains(e.target))d.open=false;}));
  document.addEventListener('keydown',e=>{if(e.key==='Escape')pops().forEach(d=>d.open=false);});
  document.querySelectorAll('.menu-list button').forEach(b=>b.addEventListener('click',()=>{$('header-menu').open=false;}));
})();

/* ===== Палитра модели: поиск по каталогу провайдера ===== */
function renderModelList(){
  const list=$('model-list');if(!list)return;
  const q=$('model-search').value.trim().toLowerCase();
  const catalog=catalogs.get($('provider').value)||[];
  const meta=new Map(catalog.map(m=>[m.id,m]));
  const ids=[...$('models').children].map(o=>o.value).filter(id=>!q||id.toLowerCase().includes(q)||String(meta.get(id)?.name||'').toLowerCase().includes(q));
  const current=$('model').value.trim();
  if(!ids.length){list.replaceChildren(node('div',$('models').children.length?'Ничего не найдено':'Нажмите «Обновить список»','muted'));return;}
  list.replaceChildren(...ids.slice(0,60).map(id=>{
    const m=meta.get(id);
    const row=node('button',undefined,'model-item'+(id===current?' on':''));row.type='button';row.setAttribute('role','option');row.setAttribute('aria-selected',id===current);
    row.append(node('span',id,'model-id'));
    if(m?.context_length)row.append(node('span',Math.round(Number(m.context_length)/1000)+'k','model-ctx'));
    row.onclick=()=>{$('model').value=id;$('model').dispatchEvent(new Event('input'));renderModelList();$('model-pop').open=false;};
    return row;
  }));
}
$('model-search').oninput=renderModelList;
$('model-pop').addEventListener('toggle',()=>{if($('model-pop').open){renderModelList();$('model-search').focus();}});
$('provider').addEventListener('change',()=>setTimeout(renderModelList,0));
