import { NextRequest, NextResponse } from 'next/server';
export const runtime = 'nodejs';
export async function POST(req: NextRequest) {
  const body = await req.json().catch(()=>({}));
  const base = process.env.NEXT_PUBLIC_FASTAPI_BASE || 'http://app:8000';
  const r = await fetch(base + '/api/agent/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await r.json();
  return NextResponse.json(data);
}
