import { NextRequest, NextResponse } from "next/server";
import { lookupItem } from "@/lib/catalog";
import type { Brand, EnrichedItem } from "@/lib/types";

const API_BASE = process.env.FASHION_API_BASE!;

const KEYS: Record<Brand, string> = {
  snitch: process.env.FASHION_API_KEY_SNITCH!,
  fashor: process.env.FASHION_API_KEY_FASHOR!,
  powerlook: process.env.FASHION_API_KEY_POWERLOOK!,
};

export async function POST(req: NextRequest): Promise<NextResponse> {
  const brand = (req.nextUrl.searchParams.get("brand") ?? "snitch") as Brand;
  const k = req.nextUrl.searchParams.get("k") ?? "8";

  const apiKey = KEYS[brand];
  if (!apiKey) return NextResponse.json({ error: "Unknown brand" }, { status: 400 });

  const formData = await req.formData();
  const image = formData.get("image") as File | null;
  if (!image) return NextResponse.json({ error: "No image provided" }, { status: 400 });

  const upstream = new FormData();
  upstream.append("image", image);

  const res = await fetch(`${API_BASE}/v1/${brand}/visual-search?k=${k}`, {
    method: "POST",
    headers: { "X-Api-Key": apiKey },
    body: upstream,
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json({ error: text }, { status: res.status });
  }

  const data = await res.json();
  const enriched: EnrichedItem[] = (data.results ?? []).map(
    (r: { item_id: string; score: number; pdp_url?: string }) => {
      const meta = lookupItem(brand, r.item_id);
      return {
        item_id: r.item_id,
        score: r.score,
        title: meta?.title ?? `Item ${r.item_id}`,
        image_url: meta?.image_url ?? "",
        price_inr: meta?.price_inr ?? 0,
        pdp_url: r.pdp_url || meta?.pdp_url || "",
        category: meta?.category ?? "",
      };
    }
  );

  return NextResponse.json({ results: enriched, request_id: data.request_id });
}
