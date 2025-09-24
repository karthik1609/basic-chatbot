import { NextRequest, NextResponse } from 'next/server';
export const runtime = 'nodejs';
export async function POST(_req: NextRequest) {
  const base = process.env.NEXT_PUBLIC_FASTAPI_BASE || 'http://app:8000';
  console.info('[frontend] POST /api/ingest ->', base + '/api/ingest');
  const r = await fetch(base + '/api/ingest', { method:'POST' });
  console.info('[frontend] /api/ingest status', r.status);
  const data = await r.json().catch(()=>({ status: r.status }));
  return NextResponse.json(data);
}
