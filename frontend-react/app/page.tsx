"use client";
import { useEffect, useMemo, useRef, useState } from 'react';
import { Markdown } from '@/components/markdown';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Wand2, Database, DatabaseBackup } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Toaster } from '@/components/ui/sonner';
import { toast } from 'sonner';

type Msg = { id: string; role: 'user'|'assistant'; content: string };
type Citation = { type: 'doc'|'sql'; tag: string; source?: string; chunk_index?: number; score?: number; rows?: number; query?: string };
type TraceEntry = Record<string, unknown>;

export default function ChatPage() {
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

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const userText = input.trim();
    setInput('');
    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', content: userText }]);
    setLoading(true);
    try {
      const res = await fetch('/api/agent', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: userText, language: 'en', session_id: sessionId }) });
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
            <Button variant="secondary" onClick={async()=>{ toast('Rebuilding index...'); const r=await fetch('/api/ingest',{method:'POST'}); const d=await r.json(); toast.success('Indexed ' + (d.chunks_indexed ?? '?') + ' chunks'); }} disabled={loading}>
              <Wand2 className="w-4 h-4 mr-1"/> Ingest
            </Button>
            <Button variant="secondary" onClick={async()=>{ toast('Initializing DB...'); const r=await fetch('/api/dbinit',{method:'POST'}); const d=await r.json(); toast.success('DB init: ' + (d.status ?? 'ok')); }} disabled={loading}>
              <Database className="w-4 h-4 mr-1"/> Init DB
            </Button>
            <Button variant="secondary" onClick={async()=>{ toast('Seeding DB...'); const r=await fetch('/api/dbseed',{method:'POST'}); const d=await r.json(); toast.success('DB seed: ' + (d.status ?? 'ok')); }} disabled={loading}>
              <DatabaseBackup className="w-4 h-4 mr-1"/> Seed DB
            </Button>
            <Button variant="secondary" onClick={onNewChat} disabled={loading}>New Chat</Button>
          </div>
        </div>
        <ScrollArea className="flex-1 pr-2">
          {messages.map(m => (
            <div key={m.id} className="mb-4">
              <div className="text-xs opacity-60 mb-1">{m.role === 'user' ? 'You' : 'Assistant'}</div>
              <div className="prose prose-invert max-w-none">
                <Markdown>{m.content}</Markdown>
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </ScrollArea>
        <form onSubmit={onSubmit} className="mt-4 flex gap-2">
          <Input value={input} onChange={(e)=>setInput(e.target.value)} placeholder="Ask anything..." disabled={loading} />
          <Button type="submit" disabled={loading}>Send</Button>
        </form>
      </Card>
      <div className="space-y-4">
        <Card className="p-4">
          <div className="font-medium mb-2">Citations</div>
          <div className="text-sm whitespace-pre-wrap">
            {citations && citations.length ? citations.map((c, i) => (
              <div key={i}>
                {c.type === 'doc' ? `[${c.tag}] ${c.source}#${c.chunk_index} (score=${(c.score||0).toFixed?.(3) ?? c.score})` : `[${c.tag}] SQL rows=${c.rows}`}
              </div>
            )) : 'No citations'}
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
