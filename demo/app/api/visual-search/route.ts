import { NextRequest, NextResponse } from "next/server";
import { lookupItem } from "@/lib/catalog";
import type { Brand, EnrichedItem } from "@/lib/types";

const API_BASE = process.env.FASHION_API_BASE!;

const KEYS: Record<Brand, string> = {
  snitch: process.env.FASHION_API_KEY_SNITCH!,
  fashor: process.env.FASHION_API_KEY_FASHOR!,
  powerlook: process.env.FASHION_API_KEY_POWERLOOK!,
};

// Bundled at build time via require() so Next.js can statically trace the imports.
const COLOR_INDEXES: Record<Brand, Record<string, { h: number; s: number; v: number }>> = {
  snitch: require("../../../public/catalog/snitch_colors.json"),
  fashor: require("../../../public/catalog/fashor_colors.json"),
  powerlook: require("../../../public/catalog/powerlook_colors.json"),
};

// Perceptual color similarity in [0, 1]. Handles achromatic colors (white/black/gray)
// by downweighting hue when saturation is low on either side.
function colorSimilarity(
  q: { h: number; s: number; v: number },
  item: { h: number; s: number; v: number }
): number {
  const hDiff = Math.min(Math.abs(q.h - item.h), 360 - Math.abs(q.h - item.h)) / 180;
  const sDiff = Math.abs(q.s - item.s);
  const vDiff = Math.abs(q.v - item.v);
  // When either side is achromatic (s < 0.15), hue comparison is noise — weight value instead.
  const achromatic = q.s < 0.15 || item.s < 0.15;
  const sim = achromatic
    ? 1 - (0.1 * hDiff + 0.2 * sDiff + 0.7 * vDiff)
    : 1 - (0.6 * hDiff + 0.3 * sDiff + 0.1 * vDiff);
  return Math.max(0, sim);
}

function hexToHSV(hex: string): { h: number; s: number; v: number } | null {
  const m = hex.match(/^([0-9a-f]{6})$/i);
  if (!m) return null;
  const r = parseInt(m[1].slice(0, 2), 16) / 255;
  const g = parseInt(m[1].slice(2, 4), 16) / 255;
  const b = parseInt(m[1].slice(4, 6), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  const v = max;
  const s = max === 0 ? 0 : d / max;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  return { h: h * 360, s, v };
}

// How much color similarity influences the final ranking (0 = off, 1 = color only).
const COLOR_WEIGHT = 0.3;

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
  const rawResults: Array<{ item_id: string; score: number; pdp_url?: string }> =
    data.results ?? [];

  // Normalize visual scores to [0, 1] within this result set so color and
  // visual similarity are on the same scale before blending.
  const maxScore = rawResults.reduce((m, r) => Math.max(m, r.score), 0) || 1;
  const minScore = rawResults.reduce((m, r) => Math.min(m, r.score), maxScore);
  const scoreRange = maxScore - minScore || 1;

  const queryHSV = hexToHSV(colorHex);
  const colorIndex = COLOR_INDEXES[brand];

  const enriched: EnrichedItem[] = rawResults
    .map((r) => {
      const meta = lookupItem(brand, r.item_id);
      const normalizedScore = (r.score - minScore) / scoreRange;

      let blendedScore = normalizedScore;
      if (queryHSV && colorIndex[r.item_id]) {
        const colSim = colorSimilarity(queryHSV, colorIndex[r.item_id]);
        blendedScore = (1 - COLOR_WEIGHT) * normalizedScore + COLOR_WEIGHT * colSim;
      }

      return {
        item_id: r.item_id,
        score: blendedScore,
        title: meta?.title ?? `Item ${r.item_id}`,
        image_url: meta?.image_url ?? "",
        price_inr: meta?.price_inr ?? 0,
        pdp_url: r.pdp_url || meta?.pdp_url || "",
        category: meta?.category ?? "",
      };
    })
    .sort((a, b) => b.score - a.score);

  return NextResponse.json({ results: enriched, request_id: data.request_id });
}
