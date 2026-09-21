'use strict';
// Text is always inserted through textContent; model output never becomes HTML.
const StudioContent = (() => {
  function blocks(text) {
    const lines = text.split('\n'), result = [];
    let plain = [], code = null;
    const flush = () => {if (plain.length) result.push({kind:'text',text:plain.join('\n')});plain=[];};
    for (const line of lines) {
      if (code) {
        if (new RegExp('^ {0,3}'+code.marker[0]+'{'+code.marker.length+',}\\s*$').test(line)) {
          result.push(code);code=null;
        } else code.text += (code.lines++ ? '\n' : '') + line;
      } else {
        const fence = line.match(/^ {0,3}(`{3,}|~{3,})([^`]*)$/);
        if (fence) {flush();code={kind:'code',marker:fence[1],language:fence[2].trim(),text:'',lines:0};}
        else plain.push(line);
      }
    }
    if (code) result.push(code);flush();
    return result;
  }
  function isDiff(text, language='') {
    return /^(diff|patch)(\s|$)/i.test(language) || /^diff --git /m.test(text) || /^--- [^\n]+\n\+\+\+ [^\n]+\n@@ /m.test(text);
  }
  function diffRows(text) {
    let old=null, next=null, inHunk=false, oldLeft=0, nextLeft=0;
    return text.split('\n').map(line => {
      const row={text:line,kind:'meta',old:'',next:''};
      const hunk=line.match(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/);
      if(hunk){old=Number(hunk[1]);next=Number(hunk[3]);oldLeft=Number(hunk[2]??1);nextLeft=Number(hunk[4]??1);inHunk=true;row.kind='hunk';}
      else if(inHunk && (oldLeft>0 || nextLeft>0) && line.startsWith('-')){row.kind='removed';row.old=old++;oldLeft--;}
      else if(inHunk && (oldLeft>0 || nextLeft>0) && line.startsWith('+')){row.kind='added';row.next=next++;nextLeft--;}
      else if(inHunk && (oldLeft>0 || nextLeft>0) && line.startsWith(' ')){row.kind='context';row.old=old++;row.next=next++;oldLeft--;nextLeft--;}
      else if(line.startsWith('diff --git ') || line.startsWith('--- ') || line.startsWith('+++ ')){row.kind='file';inHunk=false;}
      else if(!line.startsWith('\\')) inHunk=false;
      // Diff snippets without hunk headers still get colour, but no invented line numbers.
      if(row.kind==='meta' && line.startsWith('+'))row.kind='added';
      if(row.kind==='meta' && line.startsWith('-'))row.kind='removed';
      return row;
    });
  }
  function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
  function codeBlock(block) {
    const diff=isDiff(block.text,block.language),box=el('section',undefined,'code-block'),bar=el('div',undefined,'code-toolbar');
    const rows=diff?diffRows(block.text):[];
    bar.append(el('span',diff?'DIFF':block.language||'CODE'));
    if(diff)bar.append(el('span',`+${rows.filter(r=>r.kind==='added').length} −${rows.filter(r=>r.kind==='removed').length}`,'diff-count'));
    const copy=el('button','Копировать');copy.type='button';copy.setAttribute('aria-label',diff?'Копировать diff':'Копировать код');
    copy.onclick=async()=>{try{await navigator.clipboard.writeText(block.text);copy.textContent='Скопировано';}catch{copy.textContent='Не удалось скопировать';}};
    bar.append(copy);box.append(bar);
    if(diff){
      const view=el('div',undefined,'diff-view');view.tabIndex=0;view.setAttribute('role','region');view.setAttribute('aria-label','Изменения кода');
      for(const r of rows){const row=el('div',undefined,'diff-row '+r.kind);const a=el('span',r.old,'line-number'),b=el('span',r.next,'line-number');a.setAttribute('aria-hidden','true');b.setAttribute('aria-hidden','true');row.append(a,b,el('code',r.text));view.append(row);}
      box.append(view);
    }else{const pre=el('pre');pre.append(el('code',block.text));box.append(pre);}
    return box;
  }
  function render(text) {
    const root=el('div',undefined,'content');
    for(const block of blocks(text)) {
      if(block.kind==='code'||isDiff(block.text))root.append(codeBlock(block));
      else root.append(el('div',block.text,'text-block'));
    }
    return root;
  }
  return {render,blocks,diffRows,isDiff};
})();
if(typeof module!=='undefined')module.exports=StudioContent;
