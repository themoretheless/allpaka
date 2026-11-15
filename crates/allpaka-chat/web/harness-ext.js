'use strict';

const StudioHarnessExt = (() => {
  const SANDBOX_MODES = ['unsandboxed', 'read_only', 'workspace_write'];
  const PRESET_MAP = {
    readonly_audit: { sandbox_mode: 'read_only', approval_policy: 'commands', allow_writes: false },
    workspace_sandboxed: { sandbox_mode: 'workspace_write', approval_policy: 'never', allow_writes: true },
    interactive_guarded: { sandbox_mode: 'workspace_write', approval_policy: 'writes_and_commands', allow_writes: true },
    full_trusted: { sandbox_mode: 'unsandboxed', approval_policy: 'never', allow_writes: true },
  };

  let ctx = null;
  let activeTab = 'terminals';
  let activeTerminalId = '';
  let renderedInteractionKey = '';

  function $(id) {
    return ctx && ctx.$ ? ctx.$(id) : (typeof document !== 'undefined' ? document.getElementById(id) : null);
  }

  function el(tag, text, className) {
    if (ctx && ctx.node) return ctx.node(tag, text, className);
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (className) n.className = className;
    return n;
  }

  function applyPresetFields(presetId, controls = {}) {
    const presetEl = controls.presetEl ?? $('permission-preset');
    const sandboxEl = controls.sandboxEl ?? $('sandbox-mode');
    const approvalEl = controls.approvalEl ?? $('approval-policy');
    const writesEl = controls.writesEl ?? $('writes');
    if (presetEl && presetId !== undefined) presetEl.value = presetId || '';
    const spec = PRESET_MAP[presetId];
    if (!spec) return null;
    if (sandboxEl) sandboxEl.value = spec.sandbox_mode;
    if (approvalEl) approvalEl.value = spec.approval_policy;
    if (writesEl) writesEl.checked = !!spec.allow_writes;
    return { ...spec };
  }

  function applyProfileFields(profile, controls = {}) {
    if (!profile || typeof profile !== 'object') return null;
    const modeEl = controls.modeEl ?? $('mode');
    const sandboxEl = controls.sandboxEl ?? $('sandbox-mode');
    const approvalEl = controls.approvalEl ?? $('approval-policy');
    const presetEl = controls.presetEl ?? $('permission-preset');
    const writesEl = controls.writesEl ?? $('writes');
    const stepsEl = controls.stepsEl ?? $('steps');
    const teamEl = controls.teamEl ?? $('team-mode');

    const sandboxMode = SANDBOX_MODES.includes(profile.sandbox_mode) ? profile.sandbox_mode : 'unsandboxed';
    const approvalPolicy = ['never', 'commands', 'writes_and_commands', 'always'].includes(profile.approval_policy)
      ? profile.approval_policy
      : 'never';
    const maxSteps = Math.max(1, Math.min(50, Number(profile.max_steps) || 12));
    const mode = profile.mode || 'auto';
    const allowWrites = !!profile.allow_writes;
    const teamMode = !!profile.team_mode;

    if (modeEl) modeEl.value = mode;
    if (sandboxEl) sandboxEl.value = sandboxMode;
    if (approvalEl) approvalEl.value = approvalPolicy;
    // Profiles customize Settings fields cleanly; Settings.permission_preset only accepts
    // null, 'custom', or sandbox preset IDs — never a ProfileSpec id.
    if (presetEl) presetEl.value = 'custom';
    if (writesEl) {
      writesEl.checked = allowWrites;
      if (typeof writesEl.dispatchEvent === 'function' && typeof Event === 'function') {
        writesEl.dispatchEvent(new Event('change'));
      }
    }
    if (stepsEl) stepsEl.value = String(maxSteps);
    if (teamEl) teamEl.checked = teamMode;

    return {
      mode,
      sandbox_mode: sandboxMode,
      approval_policy: approvalPolicy,
      allow_writes: allowWrites,
      max_steps: maxSteps,
      team_mode: teamMode,
      permission_preset: 'custom',
    };
  }

  function parsePendingInteraction(session) {
    if (!session || typeof session !== 'object') {
      return { kind: null, approval: null, question: null };
    }
    if (session.pending_approval && session.pending_approval.id) {
      return { kind: 'approval', approval: session.pending_approval, question: null };
    }
    if (session.pending_question && session.pending_question.id) {
      return { kind: 'question', approval: null, question: session.pending_question };
    }
    return { kind: null, approval: null, question: null };
  }

  function buildApprovalPayload(requestId, decision, reason) {
    if (!['approve_once', 'approve_session', 'deny'].includes(decision)) {
      throw new Error(`Invalid approval decision: ${decision}`);
    }
    const payload = {
      request_id: String(requestId || ''),
      kind: 'approval',
      decision,
    };
    const trimmed = typeof reason === 'string' ? reason.trim() : '';
    if (trimmed) payload.reason = trimmed;
    return payload;
  }

  function buildQuestionPayload(requestId, answers, dismissReason) {
    if (dismissReason !== undefined && dismissReason !== null) {
      return {
        request_id: String(requestId || ''),
        kind: 'question',
        decision: 'dismiss',
        reason: String(dismissReason || 'Отменено пользователем'),
      };
    }
    return {
      request_id: String(requestId || ''),
      kind: 'question',
      answers: answers && typeof answers === 'object' ? answers : {},
    };
  }

  function formatTerminalList(terminals) {
    if (!Array.isArray(terminals)) return [];
    return terminals.map((t) => ({
      id: String(t.id || ''),
      title: String(t.title || t.command || t.id || 'PTY'),
      owner: String(t.owner || 'user'),
      running: t.running !== false && t.exit_code === null,
      exit_code: t.exit_code ?? null,
      label: `${t.title || t.command || t.id} [${t.owner || 'user'} · ${t.exit_code === null || t.exit_code === undefined ? 'активен' : `код ${t.exit_code}`}]`,
    }));
  }

  function formatTeamBoard(tasks) {
    if (!Array.isArray(tasks)) return [];
    return tasks.map((t) => ({
      id: String(t.id || ''),
      title: String(t.title || t.id || ''),
      status: String(t.status || 'pending'),
      assignee: t.assignee ? String(t.assignee) : '',
      dependencies: Array.isArray(t.dependencies) ? t.dependencies.map(String) : [],
      result: t.result ? String(t.result) : '',
      label: `${t.id}: ${t.title} [${t.status || 'pending'}]${t.dependencies?.length ? ` ← ${t.dependencies.join(', ')}` : ''}`,
    }));
  }

  function renderCwdBadge(session) {
    const badge = $('session-cwd-badge');
    if (!badge) return;
    const wd = session && typeof session.working_dir === 'string' ? session.working_dir.trim() : '';
    if (!wd || wd === '.') {
      badge.hidden = true;
      badge.textContent = '';
    } else {
      badge.textContent = `cwd: ${wd}`;
      badge.hidden = false;
    }
  }

  function syncSession(session) {
    renderCwdBadge(session);
    const banner = $('interaction-banner');
    if (!banner) return;

    const pending = parsePendingInteraction(session);
    if (!pending.kind) {
      renderedInteractionKey = '';
      banner.hidden = true;
      banner.replaceChildren();
      return;
    }

    const sid = session.id;
    const key = pending.kind === 'approval' ? `a:${pending.approval.id}` : `q:${pending.question.id}`;
    banner.hidden = false;
    if (key === renderedInteractionKey) return;
    renderedInteractionKey = key;
    banner.replaceChildren();

    if (pending.kind === 'approval') {
      const pa = pending.approval;
      const head = el('div', undefined, 'interaction-head');
      head.append(
        el('strong', `Требуется подтверждение: ${pa.tool}`),
        el('span', pa.summary || '', 'muted'),
      );
      banner.append(head);
      if (pa.args_preview) {
        const pre = el('pre', pa.args_preview, 'interaction-preview');
        banner.append(pre);
      }
      const controls = el('div', undefined, 'interaction-actions');
      const reasonInput = el('input', undefined, 'interaction-reason');
      reasonInput.type = 'text';
      reasonInput.placeholder = 'Комментарий или причина (необязательно)';
      controls.append(reasonInput);

      const submit = async (decision) => {
        try {
          const payload = buildApprovalPayload(pa.id, decision, reasonInput.value);
          await ctx.api(`sessions/${sid}/interaction`, payload);
          renderedInteractionKey = '';
          banner.hidden = true;
          banner.replaceChildren();
          if (ctx.reloadSession) await ctx.reloadSession();
        } catch (err) {
          if (ctx.fail) ctx.fail(err);
        }
      };

      const btnOnce = el('button', 'Разрешить 1 раз', 'primary');
      btnOnce.type = 'button';
      btnOnce.onclick = () => submit('approve_once');

      const btnSession = el('button', 'Разрешить в сессии');
      btnSession.type = 'button';
      btnSession.onclick = () => submit('approve_session');

      const btnDeny = el('button', 'Отклонить', 'danger');
      btnDeny.type = 'button';
      btnDeny.onclick = () => submit('deny');

      controls.append(btnOnce, btnSession, btnDeny);
      banner.append(controls);
      return;
    }

    if (pending.kind === 'question') {
      const pq = pending.question;
      banner.append(el('strong', 'Агент задаёт уточняющий вопрос'));
      const form = el('div', undefined, 'interaction-questions');
      const collectors = [];

      for (const q of pq.questions || []) {
        const card = el('div', undefined, 'interaction-q-card');
        card.append(el('div', q.question || q.id, 'interaction-q-text'));
        const selected = new Set();
        if (Array.isArray(q.options) && q.options.length) {
          const opts = el('div', undefined, 'interaction-q-opts');
          for (const opt of q.options) {
            const lbl = el('label', undefined, 'interaction-opt');
            const inp = el('input');
            inp.type = q.multi_select ? 'checkbox' : 'radio';
            inp.name = `pq-${pq.id}-${q.id}`;
            inp.value = opt;
            inp.onchange = () => {
              if (!q.multi_select) selected.clear();
              if (inp.checked) selected.add(opt);
              else selected.delete(opt);
            };
            lbl.append(inp, document.createTextNode(` ${opt}`));
            opts.append(lbl);
          }
          card.append(opts);
        }
        const customInput = el('input', undefined, 'interaction-reason');
        customInput.type = 'text';
        customInput.placeholder = 'Свой ответ (необязательно)';
        card.append(customInput);
        form.append(card);

        collectors.push(() => {
          const custom = customInput.value.trim();
          if (q.multi_select) {
            const vals = Array.from(selected);
            if (custom) vals.push(custom);
            return [q.id, vals];
          }
          const chosen = Array.from(selected)[0] || '';
          return [q.id, custom || chosen];
        });
      }
      banner.append(form);

      const controls = el('div', undefined, 'interaction-actions');
      const sendBtn = el('button', 'Отправить ответы', 'primary');
      sendBtn.type = 'button';
      sendBtn.onclick = async () => {
        try {
          const answers = {};
          for (const fn of collectors) {
            const [qid, val] = fn();
            answers[qid] = val;
          }
          await ctx.api(`sessions/${sid}/interaction`, buildQuestionPayload(pq.id, answers));
          renderedInteractionKey = '';
          banner.hidden = true;
          banner.replaceChildren();
          if (ctx.reloadSession) await ctx.reloadSession();
        } catch (err) {
          if (ctx.fail) ctx.fail(err);
        }
      };

      const dismissBtn = el('button', 'Пропустить');
      dismissBtn.type = 'button';
      dismissBtn.onclick = async () => {
        try {
          await ctx.api(`sessions/${sid}/interaction`, buildQuestionPayload(pq.id, {}, 'Пропущено пользователем'));
          renderedInteractionKey = '';
          banner.hidden = true;
          banner.replaceChildren();
          if (ctx.reloadSession) await ctx.reloadSession();
        } catch (err) {
          if (ctx.fail) ctx.fail(err);
        }
      };

      controls.append(sendBtn, dismissBtn);
      banner.append(controls);
    }
  }

  function reset() {
    renderedInteractionKey = '';
    activeTerminalId = '';
    const banner = $('interaction-banner');
    if (banner) {
      banner.hidden = true;
      banner.replaceChildren();
    }
    const badge = $('session-cwd-badge');
    if (badge) {
      badge.hidden = true;
      badge.textContent = '';
    }
  }

  function requireSessionId() {
    const sid = ctx && ctx.getSessionId ? ctx.getSessionId() : '';
    if (!sid) throw new Error('Сначала выберите или создайте разговор');
    return sid;
  }

  function setStatusOut(id, text) {
    const target = $(id);
    if (target) target.textContent = typeof text === 'string' ? text : JSON.stringify(text, null, 2);
  }

  async function refreshTerminals() {
    const listEl = $('ops-terminals-list');
    if (!listEl) return;
    let sid = '';
    try {
      sid = requireSessionId();
    } catch (err) {
      listEl.replaceChildren(el('p', err.message, 'muted'));
      return;
    }
    const data = await ctx.api(`sessions/${sid}/terminals`);
    const items = formatTerminalList(data.terminals);
    listEl.replaceChildren();
    if (!items.length) {
      listEl.append(el('p', 'Открытых PTY-терминалов нет.', 'muted'));
      return;
    }
    for (const item of items) {
      const row = el('div', undefined, 'ops-row');
      row.append(el('span', item.label));
      const readBtn = el('button', 'Вывод');
      readBtn.type = 'button';
      readBtn.onclick = async () => {
        activeTerminalId = item.id;
        await readActiveTerminal();
      };
      const closeBtn = el('button', 'Закрыть', 'danger');
      closeBtn.type = 'button';
      closeBtn.onclick = async () => {
        await ctx.del(`sessions/${sid}/terminals/${item.id}`);
        if (activeTerminalId === item.id) activeTerminalId = '';
        await refreshTerminals();
      };
      row.append(readBtn, closeBtn);
      listEl.append(row);
    }
    if (!activeTerminalId && items.length) {
      activeTerminalId = items[0].id;
    }
    if (activeTerminalId) await readActiveTerminal();
  }

  async function readActiveTerminal() {
    if (!activeTerminalId) return;
    const sid = requireSessionId();
    const out = await ctx.api(`sessions/${sid}/terminals/${activeTerminalId}?offset=0&max_bytes=32768`);
    const screen = $('ops-pty-output');
    if (screen) {
      const txt = out.output || '';
      screen.textContent = txt || `(PTY ${activeTerminalId}: буфер пуст, running=${out.running})`;
    }
  }

  async function refreshCheckpoints() {
    const sid = requireSessionId();
    const s = ctx.getCurrent ? ctx.getCurrent() : null;
    setStatusOut('ops-cwd-display', `working_dir: ${(s && s.working_dir) || '.'}`);

    const cpData = await ctx.api(`sessions/${sid}/checkpoints`);
    const cpBox = $('ops-checkpoints-list');
    if (cpBox) {
      cpBox.replaceChildren();
      const cps = cpData.checkpoints || [];
      if (!cps.length) cpBox.append(el('p', 'Чекпоинтов изменений пока нет.', 'muted'));
      for (const cp of cps) {
        const row = el('div', undefined, 'ops-row');
        row.append(el('span', `${cp.id} · ${cp.tool || 'patch'} (${(cp.files || []).length} файл.)`));
        const undoBtn = el('button', 'Откатить');
        undoBtn.type = 'button';
        undoBtn.onclick = async () => {
          const res = await ctx.api(`sessions/${sid}/checkpoints/${cp.id}/undo`, {});
          ctx.notify && ctx.notify(`Откат выполнен: ${(res.restored || []).length} файлов`);
          await refreshCheckpoints();
          if (ctx.reloadSession) await ctx.reloadSession();
        };
        row.append(undoBtn);
        cpBox.append(row);
      }
    }

    const spData = await ctx.api(`sessions/${sid}/spills`);
    const spBox = $('ops-spills-list');
    if (spBox) {
      spBox.replaceChildren();
      const spills = spData.spills || [];
      if (!spills.length) spBox.append(el('p', 'Сохранённых spill-файлов нет.', 'muted'));
      for (const sp of spills) {
        const row = el('div', undefined, 'ops-row');
        row.append(el('span', `${sp.spill_id} (${sp.bytes} байт)`));
        const viewBtn = el('button', 'Открыть');
        viewBtn.type = 'button';
        viewBtn.onclick = async () => {
          const content = await ctx.api(`sessions/${sid}/spills/${sp.spill_id}?offset=0&limit=16000`);
          setStatusOut('ops-spill-content', content.content || '');
        };
        row.append(viewBtn);
        spBox.append(row);
      }
    }
  }

  async function refreshWorktreesAndPr() {
    const sid = requireSessionId();
    const wtData = await ctx.api(`sessions/${sid}/worktrees`);
    const wtBox = $('ops-worktrees-list');
    if (wtBox) {
      wtBox.replaceChildren();
      const wts = wtData.worktrees || [];
      if (!wts.length) wtBox.append(el('p', wtData.raw || 'Worktrees не найдены.', 'muted'));
      for (const wt of wts) {
        wtBox.append(el('div', `${wt.path || ''} (${wt.branch || 'detached'})`, 'ops-row'));
      }
    }
    try {
      const pr = await ctx.api(`sessions/${sid}/forge-pr`);
      setStatusOut('ops-forge-status', pr);
    } catch (err) {
      setStatusOut('ops-forge-status', err.message);
    }
  }

  async function refreshTeamAndAgents() {
    const sid = requireSessionId();
    const agData = await ctx.api(`sessions/${sid}/agents`);
    const agBox = $('ops-agents-list');
    if (agBox) {
      agBox.replaceChildren();
      const list = agData.agents || [];
      if (!list.length) agBox.append(el('p', 'Активных субагентов нет.', 'muted'));
      for (const a of list) {
        const card = el('div', undefined, 'ops-card');
        card.append(el('strong', `${a.name} (${a.role}) · ${a.status}`));
        if (a.answer) card.append(el('div', String(a.answer).slice(0, 200), 'muted'));
        const row = el('div', undefined, 'ops-row');
        const msgIn = el('input');
        msgIn.type = 'text';
        msgIn.placeholder = 'Сообщение субагенту';
        const sendBtn = el('button', 'Отправить');
        sendBtn.type = 'button';
        sendBtn.onclick = async () => {
          if (!msgIn.value.trim()) return;
          await ctx.api(`sessions/${sid}/agents/${a.id}/message`, { message: msgIn.value.trim(), wake: true });
          msgIn.value = '';
          await refreshTeamAndAgents();
        };
        const stopBtn = el('button', 'Прервать', 'danger');
        stopBtn.type = 'button';
        stopBtn.onclick = async () => {
          await ctx.api(`sessions/${sid}/agents/${a.id}/interrupt`, {});
          await refreshTeamAndAgents();
        };
        row.append(msgIn, sendBtn, stopBtn);
        card.append(row);
        agBox.append(card);
      }
    }

    const teamData = await ctx.api(`sessions/${sid}/team`);
    const teamBox = $('ops-team-list');
    if (teamBox) {
      teamBox.replaceChildren();
      const tasks = formatTeamBoard(teamData.tasks);
      if (!tasks.length) teamBox.append(el('p', 'Задач Team DAG пока нет.', 'muted'));
      for (const t of tasks) {
        const row = el('div', undefined, 'ops-row team-dag-row');
        row.append(el('span', t.label));
        const doneBtn = el('button', 'Готово');
        doneBtn.type = 'button';
        doneBtn.onclick = async () => {
          await ctx.api(`sessions/${sid}/team`, { action: 'update', task_id: t.id, status: 'completed' });
          await refreshTeamAndAgents();
        };
        const delBtn = el('button', 'Удалить', 'danger');
        delBtn.type = 'button';
        delBtn.onclick = async () => {
          await ctx.api(`sessions/${sid}/team`, { action: 'delete', task_id: t.id });
          await refreshTeamAndAgents();
        };
        row.append(doneBtn, delBtn);
        teamBox.append(row);
      }
    }
  }

  async function refreshWorkflows() {
    const wfData = await ctx.api('workflows');
    const wfBox = $('ops-workflows-list');
    if (!wfBox) return;
    wfBox.replaceChildren();
    const list = wfData.workflows || [];
    if (!list.length) wfBox.append(el('p', 'Сохранённых workflows нет.', 'muted'));
    for (const wf of list) {
      const row = el('div', undefined, 'ops-row');
      row.append(el('span', `${wf.id} · ${wf.name} (${(wf.stages || []).length} шагов)`));
      const runBtn = el('button', 'Запустить', 'primary');
      runBtn.type = 'button';
      runBtn.onclick = async () => {
        const sid = requireSessionId();
        const res = await ctx.api(`sessions/${sid}/workflow`, { workflow_id: wf.id });
        setStatusOut('ops-workflow-output', res);
      };
      const delBtn = el('button', 'Удалить', 'danger');
      delBtn.type = 'button';
      delBtn.onclick = async () => {
        await ctx.del(`workflows/${wf.id}`);
        await refreshWorkflows();
      };
      row.append(runBtn, delBtn);
      wfBox.append(row);
    }
  }

  async function refreshAutomation() {
    const profData = await ctx.api('profiles');
    const profBox = $('ops-profiles-list');
    if (profBox) {
      profBox.replaceChildren();
      for (const p of profData.profiles || []) {
        const row = el('div', undefined, 'ops-row');
        row.append(el('span', `${p.id} · ${p.name} (${p.mode}, ${p.sandbox_mode})`));
        const applyBtn = el('button', 'Применить');
        applyBtn.type = 'button';
        applyBtn.onclick = () => {
          applyProfileFields(p);
          ctx.notify && ctx.notify(`Применён профиль «${p.name}»`);
        };
        row.append(applyBtn);
        if (!p.builtin) {
          const delBtn = el('button', 'Удалить', 'danger');
          delBtn.type = 'button';
          delBtn.onclick = async () => {
            await ctx.del(`profiles/${p.id}`);
            await refreshAutomation();
          };
          row.append(delBtn);
        }
        profBox.append(row);
      }
    }

    const schData = await ctx.api('schedules');
    const schBox = $('ops-schedules-list');
    if (schBox) {
      schBox.replaceChildren();
      const list = schData.schedules || [];
      if (!list.length) schBox.append(el('p', 'Расписаний нет.', 'muted'));
      for (const s of list) {
        const row = el('div', undefined, 'ops-row');
        row.append(el('span', `${s.id} · ${s.name} (${s.schedule})`));
        const trigBtn = el('button', 'Запустить');
        trigBtn.type = 'button';
        trigBtn.onclick = async () => {
          const res = await ctx.api(`schedules/${s.id}/trigger`, {});
          ctx.notify && ctx.notify(`Сессия создана: ${res.session_id || 'ok'}`);
        };
        const delBtn = el('button', 'Удалить', 'danger');
        delBtn.type = 'button';
        delBtn.onclick = async () => {
          await ctx.del(`schedules/${s.id}`);
          await refreshAutomation();
        };
        row.append(trigBtn, delBtn);
        schBox.append(row);
      }
    }

    const whData = await ctx.api('webhooks');
    const whBox = $('ops-webhooks-list');
    if (whBox) {
      whBox.replaceChildren();
      const list = whData.webhooks || [];
      if (!list.length) whBox.append(el('p', 'Webhook-триггеров нет.', 'muted'));
      for (const w of list) {
        const row = el('div', undefined, 'ops-row');
        row.append(el('span', `${w.id} · ${w.name} (вызовов: ${w.trigger_count || 0})`));
        const delBtn = el('button', 'Удалить', 'danger');
        delBtn.type = 'button';
        delBtn.onclick = async () => {
          await ctx.del(`webhooks/${w.id}`);
          await refreshAutomation();
        };
        row.append(delBtn);
        whBox.append(row);
      }
    }

    const shareBox = $('ops-shares-list');
    if (shareBox) {
      shareBox.replaceChildren();
      const sid = ctx && ctx.getSessionId ? ctx.getSessionId() : '';
      if (!sid) {
        shareBox.append(el('p', 'Выберите разговор для LiveShare-ссылок.', 'muted'));
      } else {
        const shData = await ctx.api(`sessions/${sid}/shares`);
        const list = shData.shares || [];
        if (!list.length) shareBox.append(el('p', 'LiveShare-ссылок нет.', 'muted'));
        for (const sh of list) {
          const row = el('div', undefined, 'ops-row');
          row.append(el('span', `${sh.id} · ${sh.label} (${sh.revoked ? 'отозвана' : 'активна'})`));
          if (!sh.revoked) {
            const revBtn = el('button', 'Отозвать', 'danger');
            revBtn.type = 'button';
            revBtn.onclick = async () => {
              await ctx.del(`sessions/${sid}/shares/${sh.id}`);
              await refreshAutomation();
            };
            row.append(revBtn);
          }
          shareBox.append(row);
        }
      }
    }
  }

  async function refreshActiveTab() {
    try {
      if (activeTab === 'terminals') await refreshTerminals();
      else if (activeTab === 'checkpoints') await refreshCheckpoints();
      else if (activeTab === 'worktrees') await refreshWorktreesAndPr();
      else if (activeTab === 'team') await refreshTeamAndAgents();
      else if (activeTab === 'workflows') await refreshWorkflows();
      else if (activeTab === 'automation') await refreshAutomation();
    } catch (err) {
      setStatusOut('ops-error', err.message || String(err));
    }
  }

  function switchTab(tab) {
    activeTab = tab;
    setStatusOut('ops-error', '');
    if (typeof document !== 'undefined') {
      for (const btn of document.querySelectorAll('[data-ops-tab]')) {
        btn.classList.toggle('active', btn.dataset.opsTab === tab);
      }
      for (const pane of document.querySelectorAll('[data-ops-pane]')) {
        pane.hidden = pane.dataset.opsPane !== tab;
      }
    }
    refreshActiveTab();
  }

  function wireOpsHandlers() {
    const presetSel = $('permission-preset');
    if (presetSel) {
      presetSel.onchange = () => {
        if (presetSel.value && presetSel.value !== 'custom') {
          applyPresetFields(presetSel.value);
        }
      };
    }
    const markCustom = () => {
      if (presetSel) presetSel.value = 'custom';
    };
    if ($('sandbox-mode')) $('sandbox-mode').onchange = markCustom;
    if ($('approval-policy')) $('approval-policy').onchange = markCustom;

    const dlg = $('ops-dialog');
    const openBtn = $('ops-open');
    const closeBtn = $('ops-close');
    if (openBtn && dlg) {
      openBtn.onclick = () => {
        if (typeof dlg.showModal === 'function') dlg.showModal();
        switchTab(activeTab);
      };
    }
    if (closeBtn && dlg) closeBtn.onclick = () => dlg.close();

    if (typeof document !== 'undefined') {
      for (const btn of document.querySelectorAll('[data-ops-tab]')) {
        btn.onclick = () => switchTab(btn.dataset.opsTab);
      }
    }

    const bind = (id, fn) => {
      const b = $(id);
      if (b) {
        b.onclick = async () => {
          setStatusOut('ops-error', '');
          try {
            await fn();
          } catch (err) {
            setStatusOut('ops-error', err.message || String(err));
          }
        };
      }
    };

    bind('ops-pty-open', async () => {
      const sid = requireSessionId();
      const cmd = $('ops-pty-cmd')?.value.trim();
      const res = await ctx.api(`sessions/${sid}/terminals`, {
        title: cmd || 'Shell',
        command: cmd || undefined,
        cols: 100,
        rows: 30,
      });
      activeTerminalId = res.id || activeTerminalId;
      await refreshTerminals();
    });
    bind('ops-pty-read', async () => {
      await readActiveTerminal();
    });
    bind('ops-pty-send', async () => {
      const sid = requireSessionId();
      if (!activeTerminalId) throw new Error('Нет активного PTY-терминала');
      const raw = $('ops-pty-input')?.value || '';
      await ctx.api(`sessions/${sid}/terminals/${activeTerminalId}/input`, { input: raw + '\n' });
      if ($('ops-pty-input')) $('ops-pty-input').value = '';
      await readActiveTerminal();
    });
    bind('ops-pty-resize', async () => {
      const sid = requireSessionId();
      if (!activeTerminalId) throw new Error('Нет активного PTY-терминала');
      await ctx.api(`sessions/${sid}/terminals/${activeTerminalId}/resize`, { cols: 120, rows: 36 });
    });

    bind('ops-wt-create', async () => {
      const sid = requireSessionId();
      const slug = $('ops-wt-slug')?.value.trim();
      const branch = $('ops-wt-branch')?.value.trim();
      await ctx.api(`sessions/${sid}/worktree`, {
        action: 'create',
        slug: slug || undefined,
        branch: branch || undefined,
        attach_project: !!$('ops-wt-attach')?.checked,
      });
      await refreshWorktreesAndPr();
    });
    bind('ops-wt-remove', async () => {
      const sid = requireSessionId();
      const slug = $('ops-wt-slug')?.value.trim();
      if (!slug) throw new Error('Укажите slug worktree для удаления');
      await ctx.api(`sessions/${sid}/worktree`, { action: 'remove', slug });
      await refreshWorktreesAndPr();
    });
    bind('ops-pr-checks', async () => {
      const sid = requireSessionId();
      const res = await ctx.api(`sessions/${sid}/forge-pr`, { action: 'checks' });
      setStatusOut('ops-forge-status', res);
    });
    bind('ops-pr-create', async () => {
      const sid = requireSessionId();
      const title = $('ops-pr-title')?.value.trim();
      const body = $('ops-pr-body')?.value || '';
      const base = $('ops-pr-base')?.value.trim();
      const res = await ctx.api(`sessions/${sid}/forge-pr`, {
        action: 'create',
        title,
        body,
        base: base || undefined,
      });
      setStatusOut('ops-forge-status', res);
    });

    bind('ops-team-add', async () => {
      const sid = requireSessionId();
      const title = $('ops-team-title')?.value.trim();
      const id = $('ops-team-id')?.value.trim();
      const deps = ($('ops-team-deps')?.value || '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      await ctx.api(`sessions/${sid}/team`, {
        action: 'create',
        ...(id ? { id } : {}),
        title,
        dependencies: deps,
      });
      if ($('ops-team-title')) $('ops-team-title').value = '';
      await refreshTeamAndAgents();
    });

    bind('ops-wf-save', async () => {
      const name = $('ops-wf-name')?.value.trim();
      const stagesRaw = $('ops-wf-stages')?.value.trim() || '[]';
      const stages = JSON.parse(stagesRaw);
      await ctx.api('workflows', { name, stages });
      await refreshWorkflows();
    });
    bind('ops-doc-extract', async () => {
      const path = $('ops-doc-path')?.value.trim();
      const sid = requireSessionId();
      const res = await ctx.api('documents/extract', { session_id: sid, path });
      setStatusOut('ops-workflow-output', res);
    });
    bind('ops-ssh-test', async () => {
      const host = $('ops-ssh-host')?.value.trim();
      const user = $('ops-ssh-user')?.value.trim();
      const remote_root = $('ops-ssh-root')?.value.trim() || '/tmp';
      const res = await ctx.api('ssh/test', {
        ssh: { host, user: user || undefined, remote_root },
        path: remote_root,
      });
      setStatusOut('ops-workflow-output', res);
    });

    bind('ops-bundle-export', async () => {
      const pid = ctx.getProjectId ? ctx.getProjectId() : 'default';
      const bundle = await ctx.api(`bundles/export?project_id=${encodeURIComponent(pid)}`);
      if ($('ops-bundle-json')) $('ops-bundle-json').value = JSON.stringify(bundle, null, 2);
    });
    bind('ops-bundle-import', async () => {
      const raw = $('ops-bundle-json')?.value.trim();
      if (!raw) throw new Error('Вставьте JSON бандла');
      const res = await ctx.api('bundles/import', JSON.parse(raw));
      setStatusOut('ops-automation-output', res);
      await refreshAutomation();
    });
    bind('ops-harness-reload', async () => {
      const pid = ctx.getProjectId ? ctx.getProjectId() : 'default';
      const res = await ctx.api('harness/reload', { project_id: pid });
      setStatusOut('ops-automation-output', res);
    });

    bind('ops-sch-create', async () => {
      const pid = ctx.getProjectId ? ctx.getProjectId() : 'default';
      await ctx.api('schedules', {
        name: $('ops-sch-name')?.value.trim(),
        schedule: $('ops-sch-cron')?.value.trim(),
        prompt: $('ops-sch-prompt')?.value.trim(),
        project_id: pid,
      });
      await refreshAutomation();
    });
    bind('ops-wh-create', async () => {
      const pid = ctx.getProjectId ? ctx.getProjectId() : 'default';
      const res = await ctx.api('webhooks', {
        name: $('ops-wh-name')?.value.trim(),
        prompt_template: $('ops-wh-template')?.value.trim(),
        project_id: pid,
      });
      setStatusOut('ops-automation-output', {
        webhook: res.webhook,
        secret_once: res.secret,
      });
      await refreshAutomation();
    });
    bind('ops-share-create', async () => {
      const sid = requireSessionId();
      const label = $('ops-share-label')?.value.trim() || 'Read-only snapshot';
      const res = await ctx.api(`sessions/${sid}/shares`, { label });
      setStatusOut('ops-automation-output', res);
      await refreshAutomation();
    });
    bind('ops-otlp-preview', async () => {
      const tid = $('ops-otlp-trace-id')?.value.trim();
      if (!tid) throw new Error('Укажите Trace ID');
      const res = await ctx.api(`observability/traces/${encodeURIComponent(tid)}/otlp`);
      setStatusOut('ops-automation-output', res);
    });
    bind('ops-otlp-export', async () => {
      const tid = $('ops-otlp-trace-id')?.value.trim();
      const endpoint = $('ops-otlp-endpoint')?.value.trim();
      if (!tid || !endpoint) throw new Error('Укажите Trace ID и endpoint OTLP');
      const res = await ctx.api(`observability/traces/${encodeURIComponent(tid)}/export`, { endpoint });
      setStatusOut('ops-automation-output', res);
    });
  }

  function init(deps) {
    ctx = deps;
    wireOpsHandlers();
  }

  return {
    init,
    syncSession,
    reset,
    applyPresetFields,
    applyProfileFields,
    parsePendingInteraction,
    buildApprovalPayload,
    buildQuestionPayload,
    formatTeamBoard,
    formatTerminalList,
  };
})();
