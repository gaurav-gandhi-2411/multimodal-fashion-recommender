export type Brand = "snitch" | "fashor" | "powerlook";

export interface EnrichedItem {
  item_id: string;
  score: number;
  title: string;
  image_url: string;
  price_inr: number;
  pdp_url: string;
  category: string;
  slot?: string;
}

export interface CatalogEntry {
  title: string;
  image_url: string;
  price_inr: number;
  pdp_url: string;
  category: string;
}

export const BRAND_META: Record<Brand, { label: string; tagline: string; color: string }> = {
  snitch: {
    label: "Snitch",
    tagline: "Men's Streetwear",
    color: "bg-zinc-900 text-white",
  },
  fashor: {
    label: "Fashor",
    tagline: "Women's Ethnic & Fusion",
    color: "bg-rose-600 text-white",
  },
  powerlook: {
    label: "Powerlook",
    tagline: "Men's Smart Casuals",
    color: "bg-blue-700 text-white",
  },
};
