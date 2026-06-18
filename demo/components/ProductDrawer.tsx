"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import type { EnrichedItem, Brand } from "@/lib/types";
import { ProductCard } from "./ProductCard";
import { Spinner } from "./Spinner";

interface ProductDrawerProps {
  item: EnrichedItem | null;
  brand: Brand;
  onClose: () => void;
}

export function ProductDrawer({ item, brand, onClose }: ProductDrawerProps) {
  const [similarItems, setSimilarItems] = useState<EnrichedItem[]>([]);
  const [outfitItems, setOutfitItems] = useState<EnrichedItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!item) return;
    setSimilarItems([]);
    setOutfitItems([]);
    setLoading(true);

    const base = `/api`;
    Promise.all([
      fetch(`${base}/similar?brand=${brand}&id=${item.item_id}&k=6`).then((r) => r.json()),
      fetch(`${base}/complete?brand=${brand}&id=${item.item_id}&k=4`).then((r) => r.json()),
    ])
      .then(([sim, outfit]) => {
        setSimilarItems(sim.results ?? []);
        setOutfitItems(outfit.enabled ? (outfit.results ?? []) : []);
      })
      .finally(() => setLoading(false));
  }, [item, brand]);

  const title = item?.title.replace(/\s*\([^)]+\)\s*$/, "").trim() ?? "";

  return (
    <>
      {/* Overlay */}
      <div
        className={`fixed inset-0 bg-black/30 z-40 transition-opacity duration-300 ${item ? "opacity-100" : "opacity-0 pointer-events-none"}`}
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className={`fixed top-0 right-0 h-full w-full max-w-xl bg-white z-50 shadow-2xl flex flex-col transition-transform duration-300 ease-in-out ${item ? "translate-x-0" : "translate-x-full"}`}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-5 border-b border-zinc-100">
          <div className="flex gap-4 items-start">
            {item?.image_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={item.image_url}
                alt={title}
                className="w-16 h-16 object-cover rounded-lg border border-zinc-100"
              />
            )}
            <div>
              <p className="font-semibold text-zinc-900 text-sm line-clamp-2">{title}</p>
              <p className="text-xs text-zinc-500 mt-0.5">{item?.category}</p>
              <p className="text-sm font-bold text-zinc-900 mt-1">
                ₹{item?.price_inr.toLocaleString("en-IN")}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-zinc-100 transition-colors text-zinc-500 hover:text-zinc-900 flex-shrink-0"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-8">
          {loading ? (
            <div className="flex items-center justify-center h-40 text-zinc-400">
              <Spinner size={28} />
            </div>
          ) : (
            <>
              {/* Similar items */}
              <section>
                <h3 className="text-sm font-semibold text-zinc-900 mb-3 flex items-center gap-2">
                  <span className="w-1.5 h-4 bg-zinc-900 rounded-full inline-block" />
                  More Like This
                </h3>
                {similarItems.length === 0 ? (
                  <p className="text-xs text-zinc-400">No similar items found.</p>
                ) : (
                  <div className="grid grid-cols-3 gap-3">
                    {similarItems.map((s) => (
                      <ProductCard key={s.item_id} item={s} size="sm" />
                    ))}
                  </div>
                )}
              </section>

              {/* Outfit completion */}
              {outfitItems.length > 0 && (
                <section>
                  <h3 className="text-sm font-semibold text-zinc-900 mb-3 flex items-center gap-2">
                    <span className="w-1.5 h-4 bg-rose-500 rounded-full inline-block" />
                    Complete the Outfit
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    {outfitItems.map((o) => (
                      <ProductCard key={o.item_id} item={o} size="sm" />
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
