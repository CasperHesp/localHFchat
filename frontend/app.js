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
    use_memory: JSON.parse(localStorage.getItem('use_memory') || 'false')
  },
  starters: null,   // conversation_starters.json
  intro: null,      // intro_markdown.json
  idleTimer: null,
  sessionMemory: localStorage.getItem('session_memory_text') || ''
};

function save() {
  localStorage.setItem('conversations', JSON.stringify(state.conversations));
  localStorage.setItem('currentId', state.currentId || '');
  localStorage.setItem('system_prompt', state.settings.system_prompt);
  localStorage.setItem('context_key', state.settings.context_key);
  localStorage.setItem('refine', state.settings.refine);
  localStorage.setItem('use_memory', JSON.stringify(state.settings.use_memory));
  localStorage.setItem('session_memory_text', state.sessionMemory);
}

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const currentConv = () => state.conversations.find(c => c.id === state.currentId) || null;

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

  try {
    let ctx = conv.messages.slice(-12);

    // Build system prompt (+session memory if enabled)
    const sysParts = [state.settings.system_prompt];
    if (state.settings.use_memory && state.sessionMemory.trim()) {
      sysParts.push("\n\n[Session memory]\n" + state.sessionMemory.trim());
    }
    const sysCombined = sysParts.join("");

    if (!ctx.find(m => m.role === 'system')) ctx = [{ role: 'system', content: sysCombined }, ...ctx];
    else ctx = ctx.map(m => m.role === 'system' ? {...m, content: sysCombined} : m);

    // Hidden tuned sampling
    const payload = {
      messages: ctx,
      temperature: 0.3,
      top_p: 0.95,
      repetition_penalty: 1.05,
      deterministic: false,
      max_new_tokens: 512,              // let backend auto-budget tokens
      context_key: state.settings.context_key,
      refine: state.settings.refine,     // defaults to "model"
      refine_iters: 1
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
    const data = await res.json();
    conv.messages.push({ role: 'assistant', content: data.content });
    if (!conv.title || conv.title === 'New chat') conv.title = (text.length > 30 ? text.slice(0, 30) + '…' : text);
    save(); renderChat();
  } catch (e) {
    if (e.name === 'AbortError') {
      const conv = currentConv(); if (conv) { conv.messages.push({ role: 'assistant', content: '_[stopped by user]_' }); save(); renderChat(); }
    } else {
      alert('Error: ' + e.message);
    }
  } finally {
    el('#sendBtn').disabled = false; el('#sendBtn').textContent = 'Send'; hideTyping(); currentAbort = null; resetIdleTimer();
  }
}

// --- Health & boot ---
async function checkHealthLoop() {
  const overlay = el('#loadingScreen');
  try {
    const res = await fetch('/api/health'); const j = await res.json();
    if (!j.ready) {
      overlay.classList.remove('hidden');
      setTimeout(checkHealthLoop, 1500);
      return;
    } else {
      if (sessionStorage.getItem('reloadedAfterReady') !== '1' && (overlay && !overlay.classList.contains('hidden'))) {
        sessionStorage.setItem('reloadedAfterReady', '1');
        location.reload();
        return;
      }
      overlay.classList.add('hidden');
      sessionStorage.removeItem('reloadedAfterReady');
    }
  } catch (e) {
    overlay.classList.remove('hidden');
    setTimeout(checkHealthLoop, 2500);
    return;
  }
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

  const sys = el('#sysPrompt'); const ctxSel = el('#contextKey'); const useMem = el('#useMemory');
  sys.value = state.settings.system_prompt; ctxSel.value = state.settings.context_key; useMem.checked = state.settings.use_memory;

  sys.addEventListener('input', () => { state.settings.system_prompt = sys.value; save(); resetIdleTimer(); });
  ctxSel.addEventListener('change', () => { state.settings.context_key = ctxSel.value; save(); resetIdleTimer(); });
  useMem.addEventListener('change', () => { state.settings.use_memory = useMem.checked; save(); resetIdleTimer(); });

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
  resetIdleTimer();
  checkHealthLoop();
}
boot();
