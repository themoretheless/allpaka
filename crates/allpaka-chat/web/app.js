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
  const provider=node('select');provider.dataset.swarm='provider';provider.setAttribute('aria-label','Провайдер участника');
  provider.replaceChildren(...[{id:'',name:'провайдер…'}].concat(providerList()).map(p=>{const o=node('option',p.name||p.id);o.value=p.id;return o;}));
  provider.value=providerList().some(p=>p.id===m.provider)?m.provider:'';
  const model=node('input');model.placeholder='Model ID';model.value=m.model;model.maxLength=200;model.dataset.swarm='model';model.setAttribute('aria-label','Модель участника');
  const role=node('input');role.placeholder='Роль: что именно проверяет этот агент';role.value=m.role;role.maxLength=4000;role.dataset.swarm='role';role.setAttribute('aria-label','Роль участника');
  const remove=node('button','×','swarm-remove');remove.type='button';remove.title='Убрать участника';
  remove.onclick=()=>{row.remove();updateSwarmCost();savePrefs();};
  for(const field of [label,provider,model,role])field.oninput=field.onchange=()=>{updateSwarmCost();savePrefs();};
  row.append(label,provider,model,role,remove);
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
function settings() {return {verbosity:$('verbosity').value,project_id:$('project').value,provider:$('provider').value,model:$('model').value.trim(),mode:$('mode').value,max_steps:Number($('steps').value),max_output_tokens:Number($('output-tokens').value),auto_compact:$('auto-compact').checked,compact_threshold:$('compact-threshold').value?Number($('compact-threshold').value):24000,allow_writes:$('writes').checked,json_mode:$('json-mode').checked,swarm:collectSwarm()};}
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
  try{
    const params=new URLSearchParams({q:$('search').value,project:$('project').value,folder:$('history-folder').value,sort:$('history-sort').value,offset:String(offset),limit:String(limit)});
    rows=await api('sessions?'+params);
  }catch(e){
    if(version!==chatListRequest)return;
    statusEl.replaceChildren(node('span','Не удалось загрузить историю: '+e.message));
    const retry=node('button','Повторить');retry.onclick=()=>listChats().catch(fail);
    statusEl.append(retry);statusEl.hidden=false;
    $('sessions').replaceChildren();renderHistoryMore(false);
    return;
  }
  if(version!==chatListRequest)return;
  statusEl.hidden=true;
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
function resetChat() {$('context-statistics-title').textContent=contextHeadline(null);$('context-statistics-body').replaceChildren();$('usage').textContent='';$('compact-summary').textContent='Контекст ещё не сжат.';$('branch-origin').hidden=true;active=null;current=null;lastMessages='';lastPlan='';$('plan-add').hidden=true;$('messages').innerHTML=welcome;$('title').textContent='Новый разговор';$('error').hidden=true;$('notice').hidden=true;$('queue').replaceChildren();$('plan').replaceChildren(node('li','План появится во время работы','muted'));$('history-folder').value='active';renderStatus('idle');listChats().catch(fail);refreshFolderCounts().catch(()=>{});bindSuggestions();}
function bindSuggestions() {document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;$('mode').value=b.dataset.mode;modeHelp();$('prompt').focus();});}
function renderStatus(status) {
  $('manage-chat').disabled=!active;const archived=!!(current&&current.folder&&current.folder!=='active');$('send').disabled=archived;$('resume').disabled=archived;
  const running=status==='running';$('compact').disabled=!active||running||archived;$('export').disabled=!active;$('status').textContent=stateNames[status]||status;$('status').className='badge '+status;
  $('send').textContent=running?'В очередь ↑':'Отправить ↑';$('send').title=archived?'Восстановите разговор из архива или корзины':'Отправить сообщение';$('steer').hidden=!running;$('send-now').hidden=!running;$('stop').hidden=!running;$('resume').hidden=!['paused','error'].includes(status);
}
async function openChat(id) {
  active=id;lastMessages='';const s=await api('sessions/'+id);if(active!==id)return;
  $('verbosity').value=s.settings.verbosity||'normal';$('auto-compact').checked=s.settings.auto_compact??true;$('compact-threshold').value=s.settings.compact_threshold??24000;current=s;$('history-folder').value=s.folder||'active';$('project').value=s.settings.project_id;$('provider').value=s.settings.provider;$('model').value=s.settings.model;$('mode').value=s.settings.mode;$('steps').value=s.settings.max_steps;$('output-tokens').value=s.settings.max_output_tokens??8192;$('writes').checked=s.settings.allow_writes;$('json-mode').checked=!!s.settings.json_mode;applySwarm(s.settings.swarm);
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
function renderSingleMessage(s,m,index,openKey,detailState){
  const div=node('article',undefined,'message '+m.role);
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
  }else{div.append(StudioContent.render(m.content||((m.tool_calls||[]).length?'Вызовы инструментов':m.truncated?'Лимит достигнут до текстового ответа. Увеличьте лимит ответа и продолжите.':s.status==='running'?'…':'Ответ не получен.')));}
  if((m.swarm||[]).length){
    const reports=node('details',undefined,'swarm-reports');
    const reportsKey=openKey();
    reports.dataset.openKey=reportsKey;
    reports.open=detailState.get(reportsKey)??!['running'].includes(s.status);
    const finished=(m.swarm||[]).filter(r=>r.status==='done').length;
    reports.append(node('summary',`Участники Swarm: ${finished}/${m.swarm.length} отчётов`));
    for(const report of m.swarm){
      const state=swarmState(report);
      const card=node('div',undefined,'swarm-report');
      const head=node('div',undefined,'swarm-report-head');
      head.append(node('b',report.label),node('small',`${report.provider} · ${report.model}${report.round>1?` · волна ${report.round}`:''}`),node('span',state.label,state.className));
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
function buildMessageNodes(s,detailState){
  const nodes=[];
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
        nodes.push(wrap);
        i=j;
        continue;
      }
    }
    let detailIndex=0;
    const openKey=()=>`${i}:${detailIndex++}`;
    nodes.push(renderSingleMessage(s,m,i,openKey,detailState));
    i++;
  }
  return nodes;
}
function render(s) {
  renderContextStatistics(s);

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
    const detailState=new Map([...pane.querySelectorAll('details[data-open-key]')].map(d=>[d.dataset.openKey,d.open]));
    pane.replaceChildren(...buildMessageNodes(s,detailState));lastMessages=serialized;if(atBottom)pane.scrollTop=pane.scrollHeight;
  }
  $('queue').replaceChildren(...s.queue.map((p,i)=>node('div',`${i+1}. ${p.text.slice(0,100)}`,'chip')),...s.steering.map(t=>node('div','Steer: '+t.slice(0,100),'chip')));
  if(s.queue.length||s.steering.length){const b=node('button','Очистить очередь');b.onclick=()=>control('clear_queue').catch(fail);$('queue').append(b);}
  renderPlan(s.plan);
  $('plan-add').hidden=s.folder!=='active';
  renderUsage(s);
}
function renderPlan(items,force){
  const sig=JSON.stringify(items);
  if(!force&&sig===lastPlan)return;
  lastPlan=sig;
  const list=$('plan');
  list.replaceChildren();
  if(!items.length){list.append(node('li','План появится во время работы','muted'));return;}
  items.forEach((p,i)=>{
    const li=node('li',undefined,p.status);
    const status=node('button',p.status==='completed'?'✓':p.status==='in_progress'?'◐':'○','plan-status');
    status.title='Изменить статус';
    status.onclick=()=>{p.status=p.status==='pending'?'in_progress':p.status==='in_progress'?'completed':'pending';savePlan(items);};
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
    del.onclick=()=>{items.splice(i,1);savePlan(items);};
    li.append(status,title,del);
    list.append(li);
  });
}
async function savePlan(items){
  if(!active)return;
  try{await api(`sessions/${active}/plan`,{steps:items});}
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
async function control(kind) {
  $('error').hidden=true;
  if(current&&current.folder&&current.folder!=='active'&&['send','send_now','steer','resume','compact'].includes(kind))throw new Error('Разговор в архиве или корзине. Восстановите его, чтобы продолжить.');
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
  await api(`sessions/${active}/actions`,{kind,text:sends?text:'',settings:(sends&&kind!=='steer')||['resume','compact'].includes(kind)?snapshot:undefined,images:sends?draftImages:[]});
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
    renderModelInfo();
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
$('project').onchange=()=>{renderContext();resetChat();savePrefs();};
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
$('new-chat').onclick=resetChat;$('search').oninput=()=>listChats().catch(fail);$('history-folder').onchange=()=>listChats().catch(fail);$('history-sort').onchange=()=>listChats().catch(fail);
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
$('plan-add').onclick=()=>{if(!current)return;const items=[...(current.plan||[])];items.push({title:'Новый шаг',status:'pending'});savePlan(items);};
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
$('analysis-display').onchange=()=>{savePresentation();document.querySelectorAll('details.reasoning').forEach(d=>{d.open=$('analysis-display').value==='expanded';});lastMessages='';if(current)render(current);};

let pluginStatusPolling=false;
setInterval(async()=>{
  if(!$('connection-dialog').open || pluginStatusPolling)return;
  pluginStatusPolling=true;
  try{await renderPlugins();}catch{}finally{pluginStatusPolling=false;}
},1500);
