"use client";
import { useEffect, useMemo, useRef, useState } from 'react';
import { Markdown } from '@/components/markdown';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Wand2, Database, DatabaseBackup } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Toaster } from '@/components/ui/sonner';
import { toast } from 'sonner';

type Msg = { id: string; role: 'user'|'assistant'; content: string };
type Citation = { type: 'doc'|'sql'; tag: string; source?: string; chunk_index?: number; score?: number; rows?: number; query?: string; text?: string };
type TraceEntry = Record<string, unknown>;

export default function ChatPage() {
  const API_BASE = (process.env.NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE as string) || 'http://localhost:8000';
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [trace, setTrace] = useState<TraceEntry[] | null>(null);
  const [decision, setDecision] = useState<string>('');
  const [assumptions, setAssumptions] = useState<string[]>([]);
  const [confidence, setConfidence] = useState<number|undefined>(undefined);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(()=>{ endRef.current?.scrollIntoView({behavior:'smooth'}); }, [messages]);

  useEffect(() => {
    let sid = window.localStorage.getItem('session_id');
    if (!sid) {
      sid = crypto.randomUUID();
      window.localStorage.setItem('session_id', sid);
    }
    setSessionId(sid);
  }, []);

  async function onSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!input.trim() || loading) return;
    const userText = input.trim();
    setInput('');
    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', content: userText }]);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/agent/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: userText, language: 'en', session_id: sessionId }) });
      const data = await res.json();
      if (data.session_id) {
        window.localStorage.setItem('session_id', data.session_id);
        setSessionId(data.session_id);
      }
      const text = data.ask ? data.ask : (data.answer || '');
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: text }]);
      setCitations(data.citations || []);
      setTrace(data.trace || null);
      setDecision(data.decision || '');
      setAssumptions(data.assumptions || []);
      setConfidence(typeof data.confidence === 'number' ? data.confidence : undefined);
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
    setCitations([]);
    setTrace(null);
    setDecision('');
    setAssumptions([]);
    setConfidence(undefined);
  }

  const decisionLine = useMemo(() => {
    const conf = typeof confidence === 'number' ? ` | confidence=${confidence.toFixed(3)}` : '';
    const assump = assumptions && assumptions.length ? `\nAssumptions: ${assumptions.join('; ')}` : '';
    return decision ? `Decision: ${decision}${conf}${assump}` : '';
  }, [decision, assumptions, confidence]);

  return (
    <div className="min-h-screen bg-background text-foreground p-4 grid grid-cols-1 md:grid-cols-3 gap-4">
      <Toaster />
      <Card className="p-4 md:col-span-2 flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm opacity-70">Session: {sessionId}</div>
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
                <div className="text-xs opacity-60 mb-1">{m.role === 'user' ? 'You' : 'Assistant'}</div>
                <div className="prose prose-invert max-w-none">
                  <Markdown citations={citations as unknown as { tag: string; text?: string; query?: string }[]}>{m.content}</Markdown>
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
      <div className="space-y-4">
        <Card className="p-4">
          <div className="font-medium mb-2">Citations</div>
          <div className="space-y-3 text-sm">
            {citations && citations.length ? citations.map((c, i) => (
              <div key={i} className="border border-border/50 rounded-md p-2">
                {c.type === 'doc' ? (
                  <div>
                    <div className="text-xs opacity-70 mb-1">[{c.tag}] {c.source}#{c.chunk_index} (score={(c.score||0).toFixed?.(3) ?? c.score})</div>
                    <blockquote className="border-l-2 pl-2 text-muted-foreground whitespace-pre-wrap">{c.text || '...'}</blockquote>
                  </div>
                ) : (
                  <div>
                    <div className="text-xs opacity-70 mb-1">[{c.tag}] SQL</div>
                    <pre className="text-xs overflow-auto"><code>{c.query || ''}</code></pre>
                  </div>
                )}
              </div>
            )) : <div className="text-sm opacity-70">No citations</div>}
          </div>
        </Card>
        <Card className="p-4">
          <div className="font-medium mb-2">Reasoning Trace</div>
          {decisionLine ? <div className="text-xs mb-2 whitespace-pre-wrap">{decisionLine}</div> : null}
          <pre className="text-xs whitespace-pre-wrap">{trace ? JSON.stringify(trace, null, 2) : 'No trace'}</pre>
        </Card>
      </div>
    </div>
  );
}
