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
  const k = req.nextUrl.searchParams.get("k") ?? "8";
  const colorHex = req.nextUrl.searchParams.get("color") ?? "";

  const apiKey = KEYS[brand];
  if (!apiKey) return NextResponse.json({ error: "Unknown brand" }, { status: 400 });

  const formData = await req.formData();
  const image = formData.get("image") as File | null;
  if (!image) return NextResponse.json({ error: "No image provided" }, { status: 400 });

  const upstream = new FormData();
  upstream.append("image", image);

  // Pass color hex to backend — color reranking now happens server-side.
  const colorParam = colorHex ? `&color=${colorHex}` : "";
  const res = await fetch(`${API_BASE}/v1/${brand}/visual-search?k=${k}${colorParam}`, {
    method: "POST",
    headers: { "X-Api-Key": apiKey },
    body: upstream,
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json({ error: formatBackendError(text) }, { status: res.status });
  }

  const data = await res.json();
  const rawResults: Array<{ item_id: string; score: number; pdp_url?: string }> =
    data.results ?? [];

  // Backend has already applied color+CLIP blending. Enrich with catalog metadata.
  const enriched: EnrichedItem[] = rawResults.map((r) => {
    const meta = lookupItem(brand, r.item_id);
    return {
      item_id: r.item_id,
      score: r.score,
      title: meta?.title ?? `Item ${r.item_id}`,
      image_url: meta?.image_url ?? "",
      price_inr: meta?.price_inr ?? 0,
      // Prefer the demo's own catalog snapshot over the backend's pdp_url: it's
      // hand-verified and independently deployable, so a stale/broken backend
      // catalog field (e.g. the snitch.com www-prefix bug) can't surface here.
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
