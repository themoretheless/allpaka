'use strict';
// Project files are plain text throughout: never execute/render source HTML.
window.StudioFiles = (() => {
  function lines(text, start, end) {
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) throw new Error('Укажите корректный диапазон строк.');
    const rows = text.split('\n');
    if (text.endsWith('\n')) rows.pop();
    if (end > rows.length) throw new Error('Диапазон выходит за пределы файла.');
    return rows.slice(start - 1, end).join('\n') + (end < rows.length || text.endsWith('\n') ? '\n' : '');
  }
  function selection(text, start, end) {
    if (end <= start) return null;
    return [text.slice(0, start).split('\n').length, text.slice(0, Math.max(start, end - 1)).split('\n').length];
  }
  function create({api, project, attach}) {
    const $ = id => document.getElementById(id), dialog = $('file-panel');
    let scope = '', folder = '.', current = null, epoch = 0, readEpoch = 0, timer, listing, reading;
    const element = (tag, text) => { const el = document.createElement(tag); el.textContent = text; return el; };
    const status = text => { $('file-status').textContent = text; };
    function clearPreview() {
      current = null; readEpoch++; reading?.abort();
      $('file-preview').value = ''; $('file-numbers').textContent = ''; $('file-name').textContent = 'Выберите файл';
      $('file-add').disabled = $('file-add-lines').disabled = true;
      $('file-line-start').value = $('file-line-end').value = '';
    }
    function cancel() { epoch++; readEpoch++; listing?.abort(); reading?.abort(); clearTimeout(timer); }
    function changed() {
      cancel(); clearPreview(); scope = project()?.id || ''; folder = '.';
      $('file-query').value = ''; $('file-list').replaceChildren();
      const roots = project()?.roots || [];
      $('file-root').replaceChildren(...roots.map(r => { const o = element('option', r.alias); o.value = r.alias; return o; }));
      if (dialog.open) load();
    }
    async function load() {
      clearTimeout(timer); const id = ++epoch; listing?.abort(); listing = new AbortController(); clearPreview();
      const root = $('file-root').value, query = $('file-query').value.trim(), path = query ? '.' : folder, projectId = scope;
      $('file-location').textContent = root ? root + '/' + (query ? 'Поиск' : folder) : 'Нет подключённых папок';
      $('file-up').disabled = Boolean(query) || folder === '.';
      $('file-list').replaceChildren(); status(root ? 'Загрузка…' : 'Подключите папку в настройках проекта.');
      if (!root) return;
      try {
        const result = await api('files/' + (query ? 'search' : 'list'), {project_id: projectId, root, path, query}, listing.signal);
        if (id !== epoch || project()?.id !== projectId || !dialog.open) return;
        for (const file of result.entries) {
          const button = element('button', (file.directory ? '▸ ' : '') + (query ? file.path : file.name));
          button.type = 'button'; button.title = root + '/' + file.path; button.className = 'file-entry';
          button.onclick = () => { if (file.directory) { folder = file.path; load(); } else preview(root, file.path); };
          button.onkeydown = event => {
            if (event.key === ' ') { event.preventDefault(); button.click(); }
            if (['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); (event.key === 'ArrowDown' ? button.nextElementSibling : button.previousElementSibling)?.focus(); }
          };
          $('file-list').append(button);
        }
        status(result.truncated ? 'Показана часть результатов: уточните запрос или откройте нужную папку.' : result.entries.length ? `${result.entries.length} элементов` : 'Ничего не найдено.');
        if (query) $('file-status').textContent += ' Скрытые и сборочные папки не включены в поиск.';
      } catch (error) { if (id === epoch && error.name !== 'AbortError') status(error.message); }
    }
    async function preview(root, path) {
      clearPreview(); const id = ++readEpoch, projectId = scope; reading = new AbortController();
      $('file-name').textContent = root + '/' + path; status('Чтение…');
      try {
        const file = await api('files/read', {project_id: projectId, root, path}, reading.signal);
        if (id !== readEpoch || scope !== projectId || project()?.id !== projectId || !dialog.open) return;
        current = {...file, projectId}; $('file-preview').value = file.text;
        const count = file.text === '' ? 1 : file.text.split('\n').length - (file.text.endsWith('\n') ? 1 : 0);
        $('file-numbers').textContent = Array.from({length: count}, (_, i) => i + 1).join('\n');
        $('file-line-start').max = $('file-line-end').max = String(count);
        $('file-add').disabled = false; status(`${file.bytes.toLocaleString('ru')} байт · ${count} строк · UTF-8`);
      } catch (error) { if (id === readEpoch && error.name !== 'AbortError') status('Предпросмотр недоступен: ' + error.message); }
    }
    function add(range) {
      if (!current || current.projectId !== project()?.id) return;
      try {
        const start = Number($('file-line-start').value), end = Number($('file-line-end').value);
        const name = current.root + '/' + current.path + (range ? `:L${start}-L${end}` : '');
        const text = range ? lines(current.text, start, end) : current.text;
        attach({name, text, projectId: current.projectId, sourceKey: name});
        status('Добавлено в контекст следующего сообщения: ' + name);
      } catch (error) { status(error.message); }
    }
    $('file-preview').onscroll = () => { $('file-numbers').scrollTop = $('file-preview').scrollTop; };
    function selected() {
      const preview = $('file-preview'), range = selection(preview.value, preview.selectionStart, preview.selectionEnd);
      if (range) { $('file-line-start').value = range[0]; $('file-line-end').value = range[1]; $('file-add-lines').disabled = false; }
    }
    $('file-preview').onselect = selected; $('file-preview').onkeyup = selected; $('file-preview').onmouseup = selected;
    for (const id of ['file-line-start', 'file-line-end']) $(id).oninput = () => { $('file-add-lines').disabled = !current; };
    $('file-add').onclick = () => add(false); $('file-add-lines').onclick = () => add(true);
    $('file-close').onclick = () => dialog.close(); dialog.addEventListener('close', () => { cancel(); clearPreview(); });
    $('file-root').onchange = () => { folder = '.'; load(); };
    $('file-up').onclick = () => { folder = folder.includes('/') ? folder.slice(0, folder.lastIndexOf('/')) : '.'; load(); };
    $('file-query').oninput = () => { epoch++; listing?.abort(); clearPreview(); clearTimeout(timer); timer = setTimeout(load, 180); };
    function open() { if (!dialog.open) dialog.showModal(); changed(); $('file-query').focus(); }
    document.addEventListener('keydown', event => {
      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && !event.altKey && event.key.toLowerCase() === 'p' && ![...document.querySelectorAll('dialog[open]')].some(d => d !== dialog)) { event.preventDefault(); open(); }
    });
    $('file-toggle').onclick = open;
    return {changed};
  }
  return {create, lines, selection};
})();
