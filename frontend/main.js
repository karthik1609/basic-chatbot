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

// Point API calls to the FastAPI backend on port 8000
const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;

// Persist session id for Agentic mode
let sessionId = localStorage.getItem('session_id') || null;
function resetSession() {
  sessionId = crypto.randomUUID();
  localStorage.setItem('session_id', sessionId);
  messagesEl.innerHTML = '';
  tracePre.textContent = '';
}
if (!sessionId) resetSession();

function addMessage(text, who) {
  const wrapper = document.createElement('div');
  wrapper.className = `msg ${who}`;
  wrapper.textContent = text;
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addContext(context) {
  if (!context || context.length === 0) return;
  const meta = document.createElement('div');
  meta.className = 'meta';
  const lines = context.map(c => `Source: ${c.metadata.source} | Score: ${c.score.toFixed(3)}`).join('\n');
  meta.textContent = lines;
  messagesEl.appendChild(meta);
}

function appendTrace(obj) {
  try {
    const current = tracePre.textContent ? `${tracePre.textContent}\n` : '';
    tracePre.textContent = `${current}${JSON.stringify(obj, null, 2)}`;
    tracePre.scrollTop = tracePre.scrollHeight;
  } catch {}
}

async function sendMessage() {
  const message = promptEl.value.trim();
  if (!message) return;
  addMessage(message, 'user');
  promptEl.value = '';
  try {
    if (modeEl.value === 'agent') {
      const res = await fetch(`${API_BASE}/api/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, language: languageEl.value, session_id: sessionId })
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
      addMessage(data.answer, 'bot');
      if (data.tools) appendTrace({ tool_calls: data.tools });
    } else {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, language: languageEl.value })
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
  }
}

sendBtn.addEventListener('click', sendMessage);
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage();
});

newChatBtn.addEventListener('click', () => {
  resetSession();
  addMessage('Started a new chat session.', 'bot');
});

ingestBtn.addEventListener('click', async () => {
  ingestBtn.disabled = true;
  addMessage('Rebuilding index...', 'bot');
  try {
    const res = await fetch(`${API_BASE}/api/ingest`, { method: 'POST' });
    const data = await res.json();
    addMessage(`Indexed ${data.chunks_indexed} chunks.`, 'bot');
  } catch (e) {
    addMessage(`Error: ${e.message}`, 'bot');
  } finally {
    ingestBtn.disabled = false;
  }
});

dbInitBtn.addEventListener('click', async () => {
  dbInitBtn.disabled = true;
  addMessage('Initializing DB schema...', 'bot');
  try {
    const res = await fetch(`${API_BASE}/api/db/init`, { method: 'POST' });
    const data = await res.json();
    addMessage(`DB init: ${data.status}`, 'bot');
  } catch (e) {
    addMessage(`Error: ${e.message}`, 'bot');
  } finally {
    dbInitBtn.disabled = false;
  }
});

dbSeedBtn.addEventListener('click', async () => {
  dbSeedBtn.disabled = true;
  addMessage('Seeding DB...', 'bot');
  try {
    const res = await fetch(`${API_BASE}/api/db/seed`, { method: 'POST' });
    const data = await res.json();
    addMessage(`DB seed: ${data.status}`, 'bot');
  } catch (e) {
    addMessage(`Error: ${e.message}`, 'bot');
  } finally {
    dbSeedBtn.disabled = false;
  }
});


