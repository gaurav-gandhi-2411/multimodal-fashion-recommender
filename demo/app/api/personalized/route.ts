import { NextRequest, NextResponse } from "next/server";

const API_BASE = process.env.FASHION_API_BASE!;
const H_AND_M_KEY = process.env.FASHION_API_KEY_H_AND_M!;

export async function POST(req: NextRequest): Promise<NextResponse> {
  const { user_id, k = 8 } = await req.json();
  if (!user_id) {
    return NextResponse.json({ error: "user_id required" }, { status: 400 });
  }

  const res = await fetch(`${API_BASE}/v1/h_and_m/recommend`, {
    method: "POST",
    headers: {
      "X-Api-Key": H_AND_M_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_id, k }),
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json({ error: text }, { status: res.status });
  }

  return NextResponse.json(await res.json());
}
