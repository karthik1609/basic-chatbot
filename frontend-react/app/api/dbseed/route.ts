import { NextRequest, NextResponse } from 'next/server';
export const runtime = 'edge';
export async function POST(_req: NextRequest) {
  const base = process.env.NEXT_PUBLIC_FASTAPI_BASE || (process.env.NEXT_PUBLIC_SCHEME||'http')+'://'+(process.env.NEXT_PUBLIC_HOST||'localhost')+':'+(process.env.NEXT_PUBLIC_PORT||'8000');
  const r = await fetch(base + '/api/db/seed', { method:'POST' });
  const data = await r.json().catch(()=>({ status: r.status }));
  return NextResponse.json(data);
}
