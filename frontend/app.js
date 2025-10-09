// Tiny helpers
const el = (sel) => document.querySelector(sel);

// Default system instructions
const DEFAULT_SYS = "You are a helpful brokerage assistant, proactively helping Brainbay and their affiliates. You are an optimist by nature: when there appears to be doubt, you try to infer what could be useful and you make useful suggestions for next steps.";

// Refinement default
if (localStorage.getItem('refine') !== 'model') localStorage.setItem('refine', 'model');

let currentAbort = null;

const state = {
  conversations: JSON.parse(localStorage.getItem('conversations') || '[]'),
  currentId: localStorage.getItem('currentId') || null,
  settings: {
    system_prompt: localStorage.getItem('system_prompt') || DEFAULT_SYS,
    context_key: localStorage.getItem('context_key') || 'market',
    refine: localStorage.getItem('refine') || 'model',
    use_memory: JSON.parse(localStorage.getItem('use_memory') || 'false'),
    performance_mode: localStorage.getItem('performance_mode') || 'balanced'
  },
  starters: null,   // conversation_starters.json
  intro: null,      // intro_markdown.json
  idleTimer: null,
  sessionMemory: localStorage.getItem('session_memory_text') || '',
  profile: null,
  health: { ready: false, error: null }
};

function save() {
  localStorage.setItem('conversations', JSON.stringify(state.conversations));
  localStorage.setItem('currentId', state.currentId || '');
  localStorage.setItem('system_prompt', state.settings.system_prompt);
  localStorage.setItem('context_key', state.settings.context_key);
  localStorage.setItem('refine', state.settings.refine);
  localStorage.setItem('use_memory', JSON.stringify(state.settings.use_memory));
  localStorage.setItem('performance_mode', state.settings.performance_mode);
  localStorage.setItem('session_memory_text', state.sessionMemory);
}

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const currentConv = () => state.conversations.find(c => c.id === state.currentId) || null;

const numberFmt = new Intl.NumberFormat('en-US');

function ensureCurrentModeKey() {
  const modes = state.profile?.modes || {};
  if (!Object.keys(modes).length) return state.settings.performance_mode || 'balanced';
  const current = state.settings.performance_mode;
  if (current && modes[current]) return current;
  const fallback = (state.profile?.default_mode && modes[state.profile.default_mode])
    ? state.profile.default_mode
    : Object.keys(modes)[0];
  return fallback || 'balanced';
}

function currentModeMeta() {
  const key = ensureCurrentModeKey();
  return { key, meta: (state.profile?.modes && state.profile.modes[key]) || {} };
}

function formatModelId(id) {
  if (!id) return '—';
  const last = id.split('/').pop();
  return last ? last.replace(/_/g, ' ') : id;
}

function updateStatusUI() {
  const indicator = el('#statusIndicator');
  const statusText = el('#statusText');
  const ready = !!state.health?.ready;
  const hasError = !!state.health?.error;
  if (indicator) {
    indicator.classList.toggle('ready', ready);
    indicator.classList.toggle('error', hasError && !ready);
    indicator.title = hasError ? state.health.error : (ready ? 'Model loaded' : 'Model warming up');
  }
  if (statusText) {
    statusText.textContent = ready ? 'Model ready' : (hasError ? 'Retrying model…' : 'Loading model…');
  }

  const modelEl = el('#statusModel');
  if (modelEl) {
    const id = state.health?.model_id;
    const size = state.profile?.model_size ? `· ${state.profile.model_size}` : '';
    modelEl.textContent = id ? `${formatModelId(id)} ${size}`.trim() : '—';
  }

  const deviceEl = el('#statusDevice');
  if (deviceEl) {
    const parts = [];
    if (state.health?.device) parts.push(state.health.device);
    if (state.health?.dtype) parts.push(String(state.health.dtype));
    if (state.health?.threads) parts.push(`${state.health.threads} threads`);
    deviceEl.textContent = parts.length ? parts.join(' • ') : '—';
  }

  const { key: modeKey, meta } = currentModeMeta();
  const defaults = state.profile?.defaults || {};
  const budgetEl = el('#statusBudget');
  if (budgetEl) {
    const pieces = [];
    const tokens = meta?.max_new_tokens || defaults.max_new_tokens;
    const time = meta?.max_time_s || defaults.max_time_s;
    if (tokens) pieces.push(`${numberFmt.format(Math.round(tokens))} tokens`);
    if (time) pieces.push(`${Math.round(time)} s window`);
    if (state.profile?.max_context_tokens) pieces.push(`${numberFmt.format(state.profile.max_context_tokens)} ctx`);
    budgetEl.textContent = pieces.length ? pieces.join(' • ') : '—';
  }

  const modeHint = el('#modeHint');
  if (modeHint) {
    modeHint.textContent = meta?.description || 'Adjusts response length and latency.';
  }

  const perfSelect = el('#perfMode');
  if (perfSelect) {
    const modes = state.profile?.modes || {};
    Array.from(perfSelect.options).forEach(opt => {
      const emoji = opt.dataset.emoji || '';
      const info = modes[opt.value];
      if (info?.label) {
        opt.textContent = emoji ? `${emoji} ${info.label}` : info.label;
      }
    });
    if (perfSelect.value !== modeKey) perfSelect.value = modeKey;
  }
}

// --- Markdown renderer (simple, sanitized) ---
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
function mdToHtml(md) {
  if (!md) return '';
  let blocks = [];
  md = md.replace(/```([\s\S]*?)```/g, (_, code) => {
    const i = blocks.length;
    blocks.push('<pre><code>' + escapeHtml(code.trim()) + '</code></pre>');
    return `%%BLOCK${i}%%`;
  });
  let h = escapeHtml(md);
  h = h.replace(/^######\s?(.*)$/gm,'<h6>$1</h6>')
       .replace(/^#####\s?(.*)$/gm,'<h5>$1</h5>')
       .replace(/^####\s?(.*)$/gm,'<h4>$1</h4>')
       .replace(/^###\s?(.*)$/gm,'<h3>$1</h3>')
       .replace(/^##\s?(.*)$/gm,'<h2>$1</h2>')
       .replace(/^#\s?(.*)$/gm,'<h1>$1</h1>');
  h = h.replace(/^(?:- |\* |\+ )(.*)(\n(?:(?:- |\* |\+ ).*)+)?/gm, (m) =>
    '<ul>' + m.split(/\n/).map(l => l.replace(/^(?:- |\* |\+ )/,'').trim()).map(li => `<li>${li}</li>`).join('') + '</ul>'
  );
  h = h.replace(/^(?:\d+\. )(.*)(\n(?:(?:\d+\. ).*)+)?/gm, (m) =>
    '<ol>' + m.split(/\n/).map(l => l.replace(/^\d+\. /,'').trim()).map(li => `<li>${li}</li>`).join('') + '</ol>'
  );
  h = h.replace(/^>\s?(.*)$/gm,'<blockquote>$1</blockquote>');
  h = h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
       .replace(/\*(.+?)\*/g,'<em>$1</em>')
       .replace(/`([^`]+?)`/g,'<code>$1</code>')
       .replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.split(/\n{2,}/).map(p => /<(h\d|ul|ol|pre|blockquote)/.test(p.trim()) ? p : `<p>${p.replace(/\n/g,'<br/>')}</p>`).join('\n');
  h = h.replace(/%%BLOCK(\d+)%%/g,(_,i)=>blocks[parseInt(i,10)]||'');
  return h;
}

// --- Render & UI ---
function renderConversations() {
  const ul = el('#conversations'); ul.innerHTML = '';
  state.conversations.forEach(conv => {
    const li = document.createElement('li'); if (conv.id === state.currentId) li.classList.add('active');
    const title = document.createElement('span'); title.className = 'title'; title.textContent = conv.title || 'Untitled'; li.appendChild(title);
    const btns = document.createElement('span');
    const rename = document.createElement('button'); rename.textContent = '✎';
    const del = document.createElement('button'); del.textContent = '✕';
    btns.appendChild(rename); btns.appendChild(del); li.appendChild(btns);
    li.addEventListener('click', () => { state.currentId = conv.id; save(); renderConversations(); renderChat(); resetIdleTimer(); });
    rename.addEventListener('click', (e) => { e.stopPropagation(); const t = prompt('Rename conversation', conv.title || 'Untitled'); if (t !== null) { conv.title = t.trim(); save(); renderConversations(); } });
    del.addEventListener('click', (e) => { e.stopPropagation(); if (confirm('Delete this conversation?')) { state.conversations = state.conversations.filter(c => c.id !== conv.id); if (state.currentId === conv.id) state.currentId = state.conversations[0]?.id || null; save(); renderConversations(); renderChat(); resetIdleTimer(); } });
    ul.appendChild(li);
  });
}

function renderChat() {
  const chat = el('#chat'); chat.innerHTML = '';
  const conv = currentConv();
  if (!conv) { const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'Start a new conversation.'; chat.appendChild(empty); return; }
  conv.messages.forEach(m => {
    const div = document.createElement('div');
    div.className = `msg ${m.role}`;
    const role = document.createElement('div'); role.className = 'role'; role.textContent = m.role;
    const content = document.createElement('div'); content.className = 'content'; content.innerHTML = mdToHtml(m.content);
    div.appendChild(role); div.appendChild(content); chat.appendChild(div);
  });
  chat.scrollTop = chat.scrollHeight;
}

function newConversation() {
  const id = uid(); const conv = { id, title: 'New chat', messages: [] };
  state.conversations.unshift(conv); state.currentId = id;
  save(); renderConversations(); renderChat(); insertIntroIfAny(); resetIdleTimer();
}

function showTyping() { el('#typing')?.classList.remove('hidden'); el('#stopBtn').disabled = false; }
function hideTyping() { el('#typing')?.classList.add('hidden'); el('#stopBtn').disabled = true; }

// --- Session memory helpers ---
function compactSummary(messages) {
  const items = [];
  const last = messages.slice(-16);
  for (let i=0;i<last.length;i++) {
    const m = last[i];
    if (!m.content) continue;
    let s = m.content.replace(/\s+/g,' ').trim();
    if (s.length > 240) s = s.slice(0,237)+'…';
    items.push(`- **${m.role}:** ${s}`);
  }
  const seen = new Set(); const uniq = [];
  for (const it of items) { if (!seen.has(it)) { seen.add(it); uniq.push(it); } }
  return uniq.join('\n');
}

async function summarizeAndSave() {
  const conv = currentConv(); if (!conv) return;
  // Try backend summarize if present, else fall back to client summary
  let summary = '';
  try {
    const payload = { messages: conv.messages, context_key: state.settings.context_key, use_memory: true };
    const res = await fetch('/api/summarize', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) });
    if (res.ok) { ({ summary } = await res.json()); }
    else { summary = compactSummary(conv.messages); }
  } catch { summary = compactSummary(conv.messages); }

  state.sessionMemory += (state.sessionMemory ? "\n\n" : "") + summary;
  save();
  conv.messages.push({ role:'assistant', content: '**Session memory saved:**\n\n' + summary });
  save(); renderChat();
}

function downloadMemory() {
  const a = document.createElement('a');
  const blob = new Blob([state.sessionMemory], {type:'text/plain'});
  a.href = URL.createObjectURL(blob); a.download = 'session_memory.txt'; a.click(); URL.revokeObjectURL(a.href);
}

// --- Intros & Starters ---
function keyForContext() {
  return ({ market:'company_market', geography:'company_geography', matching:'company_matching' })[state.settings.context_key] || 'company_market';
}
function insertIntroIfAny() {
  const conv = currentConv(); if (!conv) return;
  if (conv.messages.length > 0) return;
  const k = keyForContext();
  const intro = state.intro?.[k]; if (!intro) return;
  conv.messages.push({ role: 'assistant', content: intro });
  save(); renderChat();
}
function pickStarter() {
  const arr = state.starters?.[keyForContext()];
  if (!arr || !arr.length) return null;
  const idx = Math.floor(Math.random() * arr.length);
  return arr[idx];
}
function kickstartIfIdle() {
  const conv = currentConv(); if (!conv) return;
  if (conv.messages.length > 0) return;
  const s = pickStarter(); if (!s) return;
  conv.messages.push({ role: 'assistant', content: s });
  save(); renderChat();
}
function resetIdleTimer() {
  if (state.idleTimer) clearTimeout(state.idleTimer);
  state.idleTimer = setTimeout(kickstartIfIdle, 10000);
}

// --- Chat send / stop ---
async function sendMessage(text) {
  const conv = currentConv(); if (!conv) return;
  conv.messages.push({ role: 'user', content: text }); save(); renderChat();
  el('#sendBtn').disabled = true; el('#sendBtn').textContent = '...'; showTyping();

  let assistantMsg = null;
  let reader = null;
  let finalMeta = null;
  let streamError = null;

  try {
    let ctx = conv.messages.slice(-12);

    const sysParts = [state.settings.system_prompt];
    if (state.settings.use_memory && state.sessionMemory.trim()) {
      sysParts.push("\n\n[Session memory]\n" + state.sessionMemory.trim());
    }
    const sysCombined = sysParts.join("");

    if (!ctx.find(m => m.role === 'system')) ctx = [{ role: 'system', content: sysCombined }, ...ctx];
    else ctx = ctx.map(m => m.role === 'system' ? {...m, content: sysCombined} : m);

    const { key: modeKey, meta: modeMeta } = currentModeMeta();
    if (state.settings.performance_mode !== modeKey) {
      state.settings.performance_mode = modeKey;
      save();
    }
    const defaults = state.profile?.defaults || {};
    const temperature = modeMeta?.temperature ?? defaults.temperature ?? 0.3;
    const topP = modeMeta?.top_p ?? defaults.top_p ?? 0.95;
    const repetition = modeMeta?.repetition_penalty ?? defaults.repetition_penalty ?? 1.05;
    const noRepeat = modeMeta?.no_repeat_ngram_size ?? defaults.no_repeat_ngram_size ?? 3;
    const maxTime = modeMeta?.max_time_s ?? defaults.max_time_s;

    const payload = {
      messages: ctx,
      temperature,
      top_p: topP,
      repetition_penalty: repetition,
      deterministic: false,
      max_new_tokens: null,
      max_time_s: maxTime,
      no_repeat_ngram_size: noRepeat,
      context_key: state.settings.context_key,
      refine: state.settings.refine,
      refine_iters: 1,
      performance_mode: modeKey
    };
    if (payload.max_time_s == null) delete payload.max_time_s;
    payload.no_repeat_ngram_size = Math.max(1, Math.round(payload.no_repeat_ngram_size || 3));

    assistantMsg = { role: 'assistant', content: '' };
    conv.messages.push(assistantMsg);
    save();
    renderChat();

    const chatEl = el('#chat');
    const getAssistantContentEl = () => {
      const nodes = chatEl.querySelectorAll('.msg.assistant .content');
      return nodes.length ? nodes[nodes.length - 1] : null;
    };
    let assistantContentEl = getAssistantContentEl();
    const applyAssistantContent = () => {
      if (!assistantContentEl) assistantContentEl = getAssistantContentEl();
      if (assistantContentEl) {
        assistantContentEl.innerHTML = mdToHtml(assistantMsg.content);
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    };

    currentAbort = new AbortController();
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
      signal: currentAbort.signal
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    if (!res.body) {
      throw new Error('Streaming not supported in this browser.');
    }

    reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const handlePayload = (payload) => {
      if (!payload || typeof payload !== 'object') return;
      if (payload.type === 'token' && typeof payload.token === 'string') {
        assistantMsg.content += payload.token;
        applyAssistantContent();
      } else if (payload.type === 'end') {
        if (typeof payload.content === 'string') {
          assistantMsg.content = payload.content;
        }
        finalMeta = payload;
        if (payload.mode && state.settings.performance_mode !== payload.mode) {
          state.settings.performance_mode = payload.mode;
          save();
          updateStatusUI();
        }
        applyAssistantContent();
      } else if (payload.type === 'error') {
        throw new Error(payload.error || 'Generation failed.');
      }
    };

    const processBuffer = () => {
      while (true) {
        const idx = buffer.indexOf('\n');
        if (idx < 0) break;
        const raw = buffer.slice(0, idx).replace(/\r$/, '');
        buffer = buffer.slice(idx + 1);
        if (!raw) continue;
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch (_) {
          continue;
        }
        try {
          handlePayload(payload);
        } catch (err) {
          streamError = err;
          return;
        }
        if (finalMeta) return;
      }
    };

    while (!finalMeta && !streamError) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      processBuffer();
    }

    if (!streamError) {
      buffer += decoder.decode();
      processBuffer();
    }

    if (streamError) throw streamError;
    if (!finalMeta) throw new Error('Generation ended unexpectedly.');

    if (finalMeta.stop_reason === 'timeout') {
      assistantMsg.content = assistantMsg.content.trimEnd() + '\n\n_[stopped early: timeout]_';
    } else if (finalMeta.stop_reason === 'cancelled') {
      assistantMsg.content = assistantMsg.content.trimEnd() + '\n\n_[generation cancelled]_';
    }
    applyAssistantContent();

    if (!conv.title || conv.title === 'New chat') conv.title = (text.length > 30 ? text.slice(0, 30) + '…' : text);
    save();
    renderChat();
  } catch (e) {
    if (e.name === 'AbortError') {
      if (assistantMsg) {
        assistantMsg.content = assistantMsg.content.trimEnd();
        if (assistantMsg.content) {
          assistantMsg.content += '\n\n_[stopped by user]_';
        } else {
          conv.messages.pop();
          conv.messages.push({ role: 'assistant', content: '_[stopped by user]_' });
        }
        save();
        renderChat();
      } else {
        conv.messages.push({ role: 'assistant', content: '_[stopped by user]_' });
        save();
        renderChat();
      }
    } else {
      if (assistantMsg) {
        conv.messages = conv.messages.filter(m => m !== assistantMsg);
      }
      save();
      renderChat();
      alert('Error: ' + e.message);
    }
  } finally {
    if (reader) {
      try { await reader.cancel(); } catch (_) {}
    }
    el('#sendBtn').disabled = false; el('#sendBtn').textContent = 'Send'; hideTyping(); currentAbort = null; resetIdleTimer();
  }
}

// --- Health & boot ---
async function checkHealthLoop() {
  const overlay = el('#loadingScreen');
  let delay = 4000;
  try {
    const res = await fetch('/api/health');
    const j = await res.json();
    state.health = { ...(state.health || {}), ...j, error: j.last_error || null };
    if (j.profile) {
      state.profile = j.profile;
    }
    const validMode = ensureCurrentModeKey();
    if (state.settings.performance_mode !== validMode) {
      state.settings.performance_mode = validMode;
      save();
    }
    if (!j.ready) {
      overlay?.classList.remove('hidden');
      delay = 1800;
    } else {
      overlay?.classList.add('hidden');
      delay = 10000;
    }
  } catch (e) {
    const msg = e?.message || 'Network error';
    state.health = { ...(state.health || {}), ready: false, error: msg };
    overlay?.classList.remove('hidden');
    delay = 4000;
  }
  updateStatusUI();
  setTimeout(checkHealthLoop, delay);
}

function bindUI() {
  el('#newChatBtn').addEventListener('click', () => { newConversation(); });
  el('#composer').addEventListener('submit', (e) => { e.preventDefault(); const text = el('#input').value.trim(); if (!text) return; el('#input').value=''; sendMessage(text); });
  el('#exportBtn').addEventListener('click', () => {
    const a = document.createElement('a');
    const blob = new Blob([JSON.stringify(state.conversations, null, 2)], { type: 'application/json' });
    a.href = URL.createObjectURL(blob); a.download = 'chat-history.json'; a.click(); URL.revokeObjectURL(a.href);
  });
  el('#summarizeBtn').addEventListener('click', summarizeAndSave);
  el('#downloadMemoryBtn').addEventListener('click', downloadMemory);
  el('#stopBtn').addEventListener('click', () => { if (currentAbort) currentAbort.abort(); });

  const sys = el('#sysPrompt'); const ctxSel = el('#contextKey'); const useMem = el('#useMemory'); const perfSel = el('#perfMode');
  sys.value = state.settings.system_prompt; ctxSel.value = state.settings.context_key; useMem.checked = state.settings.use_memory;
  if (perfSel) {
    const mode = ensureCurrentModeKey();
    if (state.settings.performance_mode !== mode) {
      state.settings.performance_mode = mode;
      save();
    }
    perfSel.value = mode;
  }

  sys.addEventListener('input', () => { state.settings.system_prompt = sys.value; save(); resetIdleTimer(); });
  ctxSel.addEventListener('change', () => { state.settings.context_key = ctxSel.value; save(); resetIdleTimer(); });
  useMem.addEventListener('change', () => { state.settings.use_memory = useMem.checked; save(); resetIdleTimer(); });
  perfSel?.addEventListener('change', () => {
    state.settings.performance_mode = perfSel.value;
    save();
    updateStatusUI();
  });

  // dark mode
  const toggle = el('#themeToggle');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const stored = localStorage.getItem('theme') || (prefersDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', stored);
  toggle.checked = stored === 'dark';
  toggle.addEventListener('change', () => { const t = toggle.checked ? 'dark' : 'light'; document.documentElement.setAttribute('data-theme', t); localStorage.setItem('theme', t); });

  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = JSON.parse(reader.result);
          state.conversations = data; state.currentId = data[0]?.id || null; save(); renderConversations(); renderChat(); resetIdleTimer();
        } catch (err) { alert('Import failed: ' + err.message); }
      };
      reader.readAsText(file);
    }
  });

  // hide thinking bubble if user types
  el('#input').addEventListener('input', () => hideTyping());
}

async function boot() {
  // Load starters & intro (served under /static/)
  try { const res = await fetch('/static/conversation_starters.json'); state.starters = await res.json(); } catch { state.starters = null; }
  try { const res2 = await fetch('/static/intro_markdown.json'); state.intro = await res2.json(); } catch { state.intro = null; }

  if (state.conversations.length === 0) newConversation(); else { renderConversations(); renderChat(); }
  bindUI();
  updateStatusUI();
  resetIdleTimer();
  checkHealthLoop();
}
boot();
