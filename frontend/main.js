const messagesEl = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const ingestBtn = document.getElementById('ingest');
const languageEl = document.getElementById('language');
const modeEl = document.getElementById('mode');
const dbInitBtn = document.getElementById('dbinit');
const dbSeedBtn = document.getElementById('dbseed');

// Point API calls to the FastAPI backend on port 8000
const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;

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

async function sendMessage() {
  const message = promptEl.value.trim();
  if (!message) return;
  addMessage(message, 'user');
  promptEl.value = '';
  try {
    const endpoint = modeEl.value === 'agent' ? '/api/agent/chat' : '/api/chat';
    const res = await fetch(`${API_BASE}${endpoint}`, {
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
    if (data.tools) addContext(data.tools.map(t => ({ metadata: { source: t.tool }, score: 1.0 })));
  } catch (e) {
    addMessage(`Error: ${e.message}`, 'bot');
  }
}

sendBtn.addEventListener('click', sendMessage);
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage();
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


