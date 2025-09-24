import { NextRequest, NextResponse } from 'next/server';
export const runtime = 'nodejs';
export async function POST(_req: NextRequest) {
  const base = process.env.NEXT_PUBLIC_FASTAPI_BASE || 'http://app:8000';
  console.info('[frontend] POST /api/db/init ->', base + '/api/db/init');
  const r = await fetch(base + '/api/db/init', { method:'POST' });
  console.info('[frontend] /api/db/init status', r.status);
  const data = await r.json().catch(()=>({ status: r.status }));
  return NextResponse.json(data);
}
