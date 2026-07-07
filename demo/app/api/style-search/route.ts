import { NextRequest, NextResponse } from "next/server";
import { lookupItem } from "@/lib/catalog";
import { formatBackendError } from "@/lib/backend-error";
import type { Brand, EnrichedItem } from "@/lib/types";

const API_BASE = process.env.FASHION_API_BASE!;

const KEYS: Record<Brand, string> = {
  snitch: process.env.FASHION_API_KEY_SNITCH!,
  fashor: process.env.FASHION_API_KEY_FASHOR!,
  powerlook: process.env.FASHION_API_KEY_POWERLOOK!,
};

export async function POST(req: NextRequest): Promise<NextResponse> {
  const brand = (req.nextUrl.searchParams.get("brand") ?? "snitch") as Brand;
  const k = req.nextUrl.searchParams.get("k") ?? "9";
  const text = req.nextUrl.searchParams.get("text") ?? "";
  const colorHex = req.nextUrl.searchParams.get("color") ?? "";

  if (!text.trim()) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }

  const apiKey = KEYS[brand];
  if (!apiKey) return NextResponse.json({ error: "Unknown brand" }, { status: 400 });

  const colorParam = colorHex ? `&color=${colorHex}` : "";
  const res = await fetch(
    `${API_BASE}/v1/${brand}/style-search?text=${encodeURIComponent(text)}&k=${k}${colorParam}`,
    { method: "POST", headers: { "X-Api-Key": apiKey } }
  );

  if (!res.ok) {
    const errText = await res.text();
    return NextResponse.json({ error: formatBackendError(errText) }, { status: res.status });
  }

  const data = await res.json();
  const rawResults: Array<{ item_id: string; score: number; pdp_url?: string }> =
    data.results ?? [];

  const enriched: EnrichedItem[] = rawResults.map((r) => {
    const meta = lookupItem(brand, r.item_id);
    return {
      item_id: r.item_id,
      score: r.score,
      title: meta?.title ?? `Item ${r.item_id}`,
      image_url: meta?.image_url ?? "",
      price_inr: meta?.price_inr ?? 0,
      // Prefer the demo's own catalog snapshot over the backend's pdp_url — see
      // visual-search/route.ts for why (stale/broken backend catalog field shielding).
      pdp_url: meta?.pdp_url || r.pdp_url || "",
      category: meta?.category ?? "",
    };
  });

  return NextResponse.json({
    results: enriched,
    request_id: data.request_id,
    match_confidence: data.match_confidence ?? 0,
  });
}
