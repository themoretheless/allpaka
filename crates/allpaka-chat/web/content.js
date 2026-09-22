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
      if(row.kind==='meta' && line.startsWith('+'))row.kind='added';
      if(row.kind==='meta' && line.startsWith('-'))row.kind='removed';
      return row;
    });
  }
  function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}

  const KEYWORDS = {
    rust: new Set(['fn','let','mut','pub','use','mod','struct','enum','impl','trait','match','if','else','for','while','loop','return','await','async','move','ref','self','Self','const','static','type','where','crate','super','in','as','true','false','Some','None','Ok','Err']),
    python: new Set(['def','return','if','elif','else','for','while','import','from','as','class','try','except','finally','with','lambda','True','False','None','and','or','not','in','is','async','await','pass','break','continue','raise','yield','global','nonlocal']),
    js: new Set(['const','let','var','function','return','if','else','for','while','import','from','export','default','class','new','this','await','async','try','catch','finally','throw','typeof','instanceof','true','false','null','undefined','switch','case','break','continue','delete']),
    ts: new Set(['const','let','var','function','return','if','else','for','while','import','from','export','default','class','new','this','await','async','try','catch','finally','throw','typeof','instanceof','true','false','null','undefined','switch','case','break','continue','interface','type','enum','readonly','public','private']),
    json: new Set(['true','false','null']),
    bash: new Set(['if','then','else','elif','fi','for','while','do','done','echo','export','function','case','esac','in','local','return','exit','source']),
    sql: new Set(['select','from','where','insert','into','values','update','set','delete','create','table','alter','drop','index','join','left','right','inner','outer','on','group','by','order','having','limit','and','or','not','null','is','as','primary','key','foreign','references']),
    common: new Set(['true','false','null','undefined','return','if','else','for','while','function','const','let','var','class','import','export'])
  };

  function inline(text) {
    const frag = document.createDocumentFragment();
    let rest = text;
    const re = /(\*\*([^*]+)\*\*|__([^_]+)__|`([^`]+)`|\*([^*]+)\*|_([^_]+)_|\[([^\]]+)\]\(([^)]+)\))/g;
    let last = 0, m;
    while ((m = re.exec(rest))) {
      if (m.index > last) frag.append(document.createTextNode(rest.slice(last, m.index)));
      if (m[2] !== undefined) frag.append(el('strong', m[2]));
      else if (m[3] !== undefined) frag.append(el('strong', m[3]));
      else if (m[4] !== undefined) frag.append(el('code', m[4], 'inline-code'));
      else if (m[5] !== undefined) frag.append(el('em', m[5]));
      else if (m[6] !== undefined) frag.append(el('em', m[6]));
      else if (m[7] !== undefined) {
        const url = (m[8] || '').trim();
        const a = document.createElement('a');
        a.textContent = m[7];
        if (/^(https?:\/\/|mailto:)/i.test(url)) {
          a.href = url;
          a.target = '_blank';
          a.rel = 'noopener';
        }
        frag.append(a);
      }
      last = m.index + m[0].length;
    }
    if (last < rest.length) frag.append(document.createTextNode(rest.slice(last)));
    return frag;
  }

  function renderText(text) {
    const box = el('div', undefined, 'text-block');
    const lines = text.split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*$/.test(line)) { i++; continue; }
      if (/^ {0,3}(-{3,}|\*{3,})\s*$/.test(line)) {
        box.append(el('hr')); i++;
      } else if (/^ {0,3}(-|\*|\+)\s+/.test(line)) {
        const ul = el('ul');
        while (i < lines.length && /^ {0,3}(-|\*|\+)\s+/.test(lines[i])) {
          const li = el('li');
          li.append(inline(lines[i].replace(/^ {0,3}(-|\*|\+)\s+/, '')));
          ul.append(li); i++;
        }
        box.append(ul);
      } else if (/^ {0,3}\d+\.\s+/.test(line)) {
        const ol = el('ol');
        while (i < lines.length && /^ {0,3}\d+\.\s+/.test(lines[i])) {
          const li = el('li');
          li.append(inline(lines[i].replace(/^ {0,3}\d+\.\s+/, '')));
          ol.append(li); i++;
        }
        box.append(ol);
      } else if (/^ {0,3}#{1,4}\s+/.test(line)) {
        const h = line.match(/^ {0,3}(#{1,4})\s+(.*)$/);
        const div = el('div', undefined, 'md-h' + h[1].length);
        div.append(inline(h[2]));
        box.append(div); i++;
      } else if (/^ {0,3}>\s?/.test(line)) {
        const quote = el('blockquote');
        while (i < lines.length && /^ {0,3}>\s?/.test(lines[i])) {
          const p = el('div', undefined, 'md-p');
          p.append(inline(lines[i].replace(/^ {0,3}>\s?/, '')));
          quote.append(p); i++;
        }
        box.append(quote);
      } else {
        const p = el('div', undefined, 'md-p');
        p.append(inline(line));
        box.append(p); i++;
      }
    }
    return box;
  }

  function appendWords(parent, text, lang) {
    const keys = KEYWORDS[lang] || KEYWORDS.common;
    const re = /([A-Za-z_][A-Za-z0-9_]*)/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) parent.append(document.createTextNode(text.slice(last, m.index)));
      const word = m[1];
      if (keys.has(word)) parent.append(el('span', word, 'tok-key'));
      else parent.append(document.createTextNode(word));
      last = m.index + word.length;
    }
    if (last < text.length) parent.append(document.createTextNode(text.slice(last)));
  }
  function appendPlain(parent, text, lang) {
    const re = /\b(\d+(?:\.\d+)?)\b/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) appendWords(parent, text.slice(last, m.index), lang);
      parent.append(el('span', m[1], 'tok-num'));
      last = m.index + m[1].length;
    }
    appendWords(parent, text.slice(last), lang);
  }
  function highlightCode(text, language) {
    const lang = (language || '').toLowerCase();
    const code = el('code');
    const lines = text.split('\n');
    lines.forEach((line, i) => {
      if (i) code.append(document.createTextNode('\n'));
      let rest = line;
      let last = 0;
      const re = /(\/\/.*|\/\*[\s\S]*?\*\/|#.*|'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|`(?:\\.|[^`\\])*`)/g;
      let m;
      while ((m = re.exec(rest))) {
        if (m.index > last) appendPlain(code, rest.slice(last, m.index), lang);
        const val = m[1];
        const type = /^(\/\/|\/\*|#)/.test(val) ? 'tok-com' : 'tok-str';
        code.append(el('span', val, type));
        last = m.index + val.length;
      }
      if (last < rest.length) appendPlain(code, rest.slice(last), lang);
    });
    return code;
  }
  function commandLine(line) {
    return /^(\$|›|>)\s/.test(line) ||
      /^(\.\/|\/|~\/|[A-Za-z]:\\|cargo |npm |yarn |pnpm |git |python |python3 |node |curl |wget |make |cmake |allpaka |llama |npx |docker |go |rustc |javac |java |ls |cd |cat |grep |sed |awk |mkdir |rm |cp |mv |echo |export |source )/.test(line);
  }
  function renderTerminal(text) {
    const code = el('code');
    const lines = text.split('\n');
    lines.forEach((line, i) => {
      if (i) code.append(document.createTextNode('\n'));
      const prompt = line.match(/^(\$|›|>)\s/);
      if (prompt) {
        code.append(el('span', prompt[0], 'tok-prompt'));
        code.append(document.createTextNode(line.slice(prompt[0].length)));
      } else if (/^\s*#/.test(line)) {
        code.append(el('span', line, 'tok-com'));
      } else if (/\b(error|failed|failure|fatal)\b/i.test(line)) {
        code.append(el('span', line, 'tok-err'));
      } else if (/\b(success|ok|done|complete)\b/i.test(line)) {
        code.append(el('span', line, 'tok-ok'));
      } else if (commandLine(line)) {
        code.append(el('span', line, 'tok-prompt'));
      } else {
        code.append(document.createTextNode(line));
      }
    });
    return code;
  }
  function codeBlock(block) {
    const terminal=/^(bash|sh|shell|zsh|console|terminal|cmd|powershell|fish)$/i.test(block.language||'');
    const diff=isDiff(block.text,block.language);
    const box=el('section',undefined,'code-block'+(terminal?' terminal':''));
    const bar=el('div',undefined,'code-toolbar');
    const rows=diff?diffRows(block.text):[];
    if(terminal){
      const dots=el('span',undefined,'term-dots');
      for(const color of ['#ff5f57','#febc2e','#28c840']){
        const d=el('i',undefined,'term-dot');d.style.background=color;dots.append(d);
      }
      bar.append(dots,el('span','ТЕРМИНАЛ'));
    }else{
      bar.append(el('span',diff?'DIFF':block.language||'CODE'));
    }
    if(diff)bar.append(el('span',`+${rows.filter(r=>r.kind==='added').length} −${rows.filter(r=>r.kind==='removed').length}`,'diff-count'));
    const copy=el('button','Копировать');copy.type='button';copy.setAttribute('aria-label',diff?'Копировать diff':'Копировать код');
    copy.onclick=async()=>{try{await navigator.clipboard.writeText(block.text);copy.textContent='Скопировано';}catch{copy.textContent='Не удалось скопировать';}};
    bar.append(copy);box.append(bar);
    if(diff){
      const view=el('div',undefined,'diff-view');view.tabIndex=0;view.setAttribute('role','region');view.setAttribute('aria-label','Изменения кода');
      for(const r of rows){const row=el('div',undefined,'diff-row '+r.kind);const a=el('span',r.old,'line-number'),b=el('span',r.next,'line-number');a.setAttribute('aria-hidden','true');b.setAttribute('aria-hidden','true');row.append(a,b,el('code',r.text));view.append(row);}
      box.append(view);
    }else{
      const pre=el('pre');
      pre.append(terminal ? renderTerminal(block.text) : highlightCode(block.text, block.language));
      box.append(pre);
    }
    return box;
  }
  function render(text) {
    const root=el('div',undefined,'content');
    for(const block of blocks(text)) {
      if(block.kind==='code'||isDiff(block.text))root.append(codeBlock(block));
      else root.append(renderText(block.text));
    }
    return root;
  }
  return {render,blocks,diffRows,isDiff};
})();
if(typeof module!=='undefined')module.exports=StudioContent;
