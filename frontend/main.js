document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const messagesEl = document.getElementById('messages');
  const promptEl = document.getElementById('prompt');
  const sendBtn = document.getElementById('send');
  const ingestBtn = document.getElementById('ingest');
  const languageEl = document.getElementById('language');
  const modeEl = document.getElementById('mode');
  const dbInitBtn = document.getElementById('dbinit');
  const dbSeedBtn = document.getElementById('dbseed');
  const newChatBtn = document.getElementById('newchat');
  const tracePre = document.getElementById('tracePre');
  const modelProfileEl = document.getElementById('modelProfile');

  // Point API calls to the FastAPI backend on port 8000
  const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;
  // Load model profiles
  async function loadProfiles() {
    try {
      const res = await fetch(`${API_BASE}/api/models`);
      const data = await res.json();
      const profiles = data.profiles || [];
      if (modelProfileEl) {
        modelProfileEl.innerHTML = '';
        for (const p of profiles) {
          const opt = document.createElement('option');
          opt.value = p.id;
          opt.textContent = `${p.label || p.id}`;
          modelProfileEl.appendChild(opt);
        }
        const saved = localStorage.getItem('model_profile');
        modelProfileEl.value = saved || data.default || (profiles[0] && profiles[0].id) || '';
        modelProfileEl.addEventListener('change', () => {
          localStorage.setItem('model_profile', modelProfileEl.value);
        });
      }
    } catch (e) {
      console.warn('Failed to load model profiles', e);
    }
  }
  loadProfiles();


  // UUID fallback for older browsers
  function uuid() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
    const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
    return `${Date.now()}-${s4()}-${s4()}-${s4()}-${s4()}${s4()}`;
  }

  // Persist session id for Agentic mode
  let sessionId = localStorage.getItem('session_id') || null;
  function resetSession() {
    sessionId = uuid();
    localStorage.setItem('session_id', sessionId);
    if (messagesEl) messagesEl.innerHTML = '';
    if (tracePre) tracePre.textContent = '';
  }
  if (!sessionId) resetSession();

  function addMessage(text, who) {
    if (!messagesEl) return;
    const wrapper = document.createElement('div');
    wrapper.className = `msg ${who}`;
    wrapper.textContent = text;
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addContext(context) {
    if (!context || context.length === 0 || !messagesEl) return;
    const meta = document.createElement('div');
    meta.className = 'meta';
    const lines = context.map(c => `Source: ${c.metadata.source} | Score: ${c.score.toFixed(3)}`).join('\n');
    meta.textContent = lines;
    messagesEl.appendChild(meta);
  }

  function appendTrace(obj) {
    try {
      if (!tracePre) return;
      const current = tracePre.textContent ? `${tracePre.textContent}\n` : '';
      tracePre.textContent = `${current}${JSON.stringify(obj, null, 2)}`;
      tracePre.scrollTop = tracePre.scrollHeight;
    } catch {}
  }

  function addCitations(citations) {
    if (!citations || citations.length === 0 || !messagesEl) return;
    const meta = document.createElement('div');
    meta.className = 'meta';
    const lines = citations.map(c => {
      if (c.type === 'doc') return `[${c.tag}] ${c.source}#${c.chunk_index} (score=${(c.score||0).toFixed(3)})`;
      if (c.type === 'sql') return `[${c.tag}] SQL rows=${c.rows}`;
      return JSON.stringify(c);
    }).join('\n');
    meta.textContent = `Citations:\n${lines}`;
    messagesEl.appendChild(meta);
  }

  async function sendMessage() {
    if (sendBtn) sendBtn.disabled = true;
    if (promptEl) promptEl.disabled = true;
    const message = (promptEl && promptEl.value || '').trim();
    if (!message) return;
    addMessage(message, 'user');
    if (promptEl) promptEl.value = '';
    try {
      if (modeEl && modeEl.value === 'agent') {
        const res = await fetch(`${API_BASE}/api/agent/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, language: languageEl ? languageEl.value : 'en', session_id: sessionId, model_profile: modelProfileEl ? modelProfileEl.value : undefined })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        if (data.session_id) {
          sessionId = data.session_id;
          localStorage.setItem('session_id', sessionId);
        }
        if (data.ask) {
          addMessage(data.ask, 'bot');
          appendTrace({ stage: 'ask', question: data.ask });
        } else {
          addMessage(data.answer, 'bot');
          if (data.citations) addCitations(data.citations);
          if (data.trace) appendTrace({ trace: data.trace });
          if (data.tools) appendTrace({ tool_calls: data.tools });
        }
      } else {
        const res = await fetch(`${API_BASE}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, language: languageEl ? languageEl.value : 'en', model_profile: modelProfileEl ? modelProfileEl.value : undefined })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        addMessage(data.answer, 'bot');
        if (data.context) addContext(data.context);
      }
    } catch (e) {
      addMessage(`Error: ${e.message}`, 'bot');
      console.error(e);
    } finally {
      if (sendBtn) sendBtn.disabled = false;
      if (promptEl) promptEl.disabled = false;
      if (promptEl) promptEl.focus();
    }
  }

  if (sendBtn) sendBtn.addEventListener('click', sendMessage);
  if (promptEl) {
    promptEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendMessage();
    });
  }

  if (newChatBtn) newChatBtn.addEventListener('click', () => {
    resetSession();
    addMessage('Started a new chat session.', 'bot');
  });

  if (ingestBtn) ingestBtn.addEventListener('click', async () => {
    ingestBtn.disabled = true;
    addMessage('Rebuilding index...', 'bot');
    try {
      const res = await fetch(`${API_BASE}/api/ingest`, { method: 'POST' });
      const data = await res.json();
      addMessage(`Indexed ${data.chunks_indexed} chunks.`, 'bot');
    } catch (e) {
      addMessage(`Error: ${e.message}`, 'bot');
      console.error(e);
    } finally {
      ingestBtn.disabled = false;
    }
  });

  if (dbInitBtn) dbInitBtn.addEventListener('click', async () => {
    dbInitBtn.disabled = true;
    addMessage('Initializing DB schema...', 'bot');
    try {
      const res = await fetch(`${API_BASE}/api/db/init`, { method: 'POST' });
      const data = await res.json();
      addMessage(`DB init: ${data.status}`, 'bot');
    } catch (e) {
      addMessage(`Error: ${e.message}`, 'bot');
      console.error(e);
    } finally {
      dbInitBtn.disabled = false;
    }
  });

  if (dbSeedBtn) dbSeedBtn.addEventListener('click', async () => {
    dbSeedBtn.disabled = true;
    addMessage('Seeding DB...', 'bot');
    try {
      const res = await fetch(`${API_BASE}/api/db/seed`, { method: 'POST' });
      const data = await res.json();
      addMessage(`DB seed: ${data.status}`, 'bot');
    } catch (e) {
      addMessage(`Error: ${e.message}`, 'bot');
      console.error(e);
    } finally {
      dbSeedBtn.disabled = false;
    }
  });

  console.log('Frontend initialized; API_BASE=', API_BASE, 'session_id=', sessionId);
});


