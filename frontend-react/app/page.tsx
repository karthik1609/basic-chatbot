"use client";
import { useEffect, useMemo, useRef, useState } from 'react';
import { Markdown } from '@/components/markdown';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Wand2, Database, DatabaseBackup, GitBranch } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Toaster } from '@/components/ui/sonner';
import { toast } from 'sonner';
import TraceFlow, { type RFTrace, type RFTraceEntry } from '@/components/trace-flow';

export type Citation = { type: 'doc'|'sql'; tag: string; source?: string; chunk_index?: number; score?: number; rows?: number; query?: string; text?: string };

type Msg = {
  id: string;
  role: 'user'|'assistant';
  content: string;
  citations?: Citation[];
  trace?: TraceEntry[] | null;
  decision?: string;
  assumptions?: string[];
  confidence?: number;
};
type TraceEntry = Record<string, unknown>;

export default function ChatPage() {
  const API_BASE = (process.env.NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE as string) || 'http://localhost:8000';
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [traceOpenFor, setTraceOpenFor] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [flowOpenFor, setFlowOpenFor] = useState<string | null>(null);
  const [lang, setLang] = useState<string>('en');

  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(()=>{ endRef.current?.scrollIntoView({behavior:'smooth'}); }, [messages]);

  useEffect(() => {
    let sid = window.localStorage.getItem('session_id');
    if (!sid) {
      sid = crypto.randomUUID();
      window.localStorage.setItem('session_id', sid);
    }
    setSessionId(sid);
    const savedLang = window.localStorage.getItem('lang');
    if (savedLang) setLang(savedLang);
  }, []);

  async function onSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!input.trim() || loading) return;
    const userText = input.trim();
    setInput('');
    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', content: userText }]);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/agent/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: userText, language: lang, session_id: sessionId }) });
      const data = await res.json();
      if (data.session_id) {
        window.localStorage.setItem('session_id', data.session_id);
        setSessionId(data.session_id);
      }
      const text = data.ask ? data.ask : (data.answer || '');
      const msg: Msg = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: text,
        citations: (data.citations || []) as Citation[],
        trace: (data.trace || null) as TraceEntry[] | null,
        decision: data.decision || '',
        assumptions: (data.assumptions || []) as string[],
        confidence: typeof data.confidence === 'number' ? data.confidence : undefined,
      };
      setMessages(prev => [...prev, msg]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: `Error: ${msg}` }]);
    } finally {
      setLoading(false);
    }
  }

  function onNewChat() {
    const sid = crypto.randomUUID();
    window.localStorage.setItem('session_id', sid);
    setSessionId(sid);
    setMessages([]);
    setTraceOpenFor(null);
    setFlowOpenFor(null);
  }

  const selectedMsg: Msg | undefined = useMemo(() => messages.find(m => m.id === (traceOpenFor || flowOpenFor)), [messages, traceOpenFor, flowOpenFor]);
  const decisionLine = useMemo(() => {
    if (!selectedMsg) return '';
    const conf = typeof selectedMsg.confidence === 'number' ? ` | confidence=${selectedMsg.confidence.toFixed(3)}` : '';
    const assump = selectedMsg.assumptions && selectedMsg.assumptions.length ? `\nAssumptions: ${selectedMsg.assumptions.join('; ')}` : '';
    return selectedMsg.decision ? `Decision: ${selectedMsg.decision}${conf}${assump}` : '';
  }, [selectedMsg]);

  const traceOpen = Boolean(traceOpenFor);
  const flowOpen = Boolean(flowOpenFor);
  return (
    <div className="min-h-screen bg-background text-foreground p-4">
      <Toaster />
      <div className={`mx-auto flex gap-4 md:flex-row flex-col max-w-screen-2xl`}>
        {(traceOpen || flowOpen) ? (
          <aside className="md:w-1/3 w-full">
            <Card className="p-4 h-full max-h-[calc(100vh-2rem)] overflow-auto">
              <div className="font-medium mb-2">{traceOpen ? 'Reasoning Trace' : 'Reasoning Flow'}</div>
              {decisionLine ? <div className="text-xs mb-2 whitespace-pre-wrap">{decisionLine}</div> : null}
              {traceOpen ? (
                <pre className="text-xs whitespace-pre-wrap">{selectedMsg?.trace ? JSON.stringify(selectedMsg.trace, null, 2) : 'No trace'}</pre>
              ) : (
                <div className="h-[70vh] min-h-[360px]">
                  <TraceFlow trace={(selectedMsg?.trace as unknown as RFTrace) || []} />
                </div>
              )}
            </Card>
          </aside>
        ) : null}
        <Card className={`p-4 flex flex-col ${(traceOpen || flowOpen) ? 'md:w-2/3 w-full' : 'w-full'} max-h-[calc(100vh-2rem)]`}>
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm opacity-70 flex items-center gap-3">
            <span>Session: {sessionId}</span>
            <label className="text-xs opacity-70" htmlFor="lang-select">Language</label>
            <select
              id="lang-select"
              className="bg-background border border-border rounded px-2 py-1 text-xs"
              value={lang}
              onChange={(e)=>{ const v = (e.target as HTMLSelectElement).value; setLang(v); window.localStorage.setItem('lang', v); }}
              disabled={loading}
            >
              <option value="en">English</option>
              <option value="nl">Nederlands</option>
              <option value="sv">Svenska</option>
              <option value="de">Deutsch</option>
              <option value="fr">Français</option>
              <option value="es">Español</option>
            </select>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={async()=>{ try{ toast('Rebuilding index...'); const r=await fetch(`${API_BASE}/api/ingest`,{method:'POST'}); if(!r.ok){ const t=await r.text(); throw new Error(t||`HTTP ${r.status}`);} const d=await r.json(); toast.success(`Indexed ${d.chunks_indexed ?? '?' } chunks`);}catch(e){ const err = e instanceof Error ? e.message : String(e); toast.error(`Ingest failed: ${err}`);} }} disabled={loading}>
              <Wand2 className="w-4 h-4 mr-1"/> Ingest
            </Button>
            <Button variant="secondary" onClick={async()=>{ try{ toast('Initializing DB...'); const r=await fetch(`${API_BASE}/api/db/init`,{method:'POST'}); if(!r.ok){ const t=await r.text(); throw new Error(t||`HTTP ${r.status}`);} const d=await r.json(); toast.success(`DB init: ${d.status||'ok'}`);}catch(e){ const err = e instanceof Error ? e.message : String(e); toast.error(`Init failed: ${err}`);} }} disabled={loading}>
              <Database className="w-4 h-4 mr-1"/> Init DB
            </Button>
            <Button variant="secondary" onClick={async()=>{ try{ toast('Seeding DB...'); const r=await fetch(`${API_BASE}/api/db/seed`,{method:'POST'}); if(!r.ok){ const t=await r.text(); throw new Error(t||`HTTP ${r.status}`);} const d=await r.json(); toast.success(`DB seed: ${d.status||'ok'}`);}catch(e){ const err = e instanceof Error ? e.message : String(e); toast.error(`Seed failed: ${err}`);} }} disabled={loading}>
              <DatabaseBackup className="w-4 h-4 mr-1"/> Seed DB
            </Button>
            <Button variant="secondary" onClick={onNewChat} disabled={loading}>New Chat</Button>
          </div>
        </div>
        <ScrollArea className="flex-1 pr-2">
          {messages.map(m => (
            <div key={m.id} className={`mb-4 flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] rounded-xl px-3 py-2 ${m.role === 'user' ? 'bg-primary/15 border border-primary/30' : 'bg-secondary/20 border border-secondary/30'}`}>
                <div className="text-xs opacity-60 mb-1 flex items-center justify-between gap-3">
                  <span>{m.role === 'user' ? 'You' : 'Assistant'}</span>
                  {m.role === 'assistant' ? (
                    <Button
                      type="button"
                      variant="ghost"
                      className="h-6 px-2 text-xs"
                      onClick={() => { setTraceOpenFor(traceOpenFor === m.id ? null : m.id); setFlowOpenFor(prev => prev === m.id ? null : null); }}
                    >
                      {traceOpenFor === m.id ? 'Hide reasoning' : 'Show reasoning'}
                    </Button>
                  ) : null}
                  {m.role === 'assistant' ? (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-6 px-2 text-xs border-amber-400 text-amber-300 hover:bg-amber-900/20"
                      disabled={!m.trace || (Array.isArray(m.trace) && m.trace.length === 0)}
                      onClick={() => { if (!(!m.trace || (Array.isArray(m.trace) && m.trace.length === 0))) { setFlowOpenFor(flowOpenFor === m.id ? null : m.id); setTraceOpenFor(prev => prev === m.id ? null : null); } }}
                    >
                      <GitBranch className="w-3.5 h-3.5 mr-1" /> {flowOpenFor === m.id ? 'Hide flow' : 'Show flow'}
                    </Button>
                  ) : null}
                </div>
                <div className="prose prose-invert max-w-none">
                  <Markdown citations={(m.citations || []).map(c => ({ tag: c.tag, text: c.text, query: c.query }))}>{m.content}</Markdown>
                </div>
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </ScrollArea>
        <form onSubmit={onSubmit} className="mt-4 flex gap-2 items-end">
          <Textarea
            value={input}
            onChange={(e)=>setInput(e.target.value)}
            placeholder="Ask anything..."
            disabled={loading}
            className="min-h-[48px]"
            onKeyDown={(e)=>{
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(); }
            }}
          />
          <Button type="submit" disabled={loading}>
            {loading ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-4 w-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                Waiting...
              </span>
            ) : 'Send'}
          </Button>
        </form>
        </Card>
      </div>
    </div>
  );
}
