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

export async function GET(req: NextRequest): Promise<NextResponse> {
  const brand = (req.nextUrl.searchParams.get("brand") ?? "snitch") as Brand;
  const id = req.nextUrl.searchParams.get("id");
  const k = req.nextUrl.searchParams.get("k") ?? "4";

  if (!id) return NextResponse.json({ error: "Missing id" }, { status: 400 });

  const apiKey = KEYS[brand];
  if (!apiKey) return NextResponse.json({ error: "Unknown brand" }, { status: 400 });

  const res = await fetch(`${API_BASE}/v1/${brand}/item/${id}/complete?k=${k}`, {
    headers: { "X-Api-Key": apiKey },
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json({ error: formatBackendError(text) }, { status: res.status });
  }

  const data = await res.json();

  if (!data.enabled) {
    return NextResponse.json({ results: [], enabled: false });
  }

  const enriched: EnrichedItem[] = (data.results ?? []).map(
    (r: { item_id: string; score: number; slot?: string; pdp_url?: string }) => {
      const meta = lookupItem(brand, r.item_id);
      return {
        item_id: r.item_id,
        score: r.score,
        slot: r.slot ?? "",
        title: meta?.title ?? `Item ${r.item_id}`,
        image_url: meta?.image_url ?? "",
        price_inr: meta?.price_inr ?? 0,
        pdp_url: r.pdp_url || meta?.pdp_url || "",
        category: meta?.category ?? "",
      };
    }
  );

  return NextResponse.json({ results: enriched, enabled: true });
}
