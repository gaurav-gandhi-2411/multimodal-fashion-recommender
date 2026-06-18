"use client";

import type { EnrichedItem } from "@/lib/types";

interface ProductCardProps {
  item: EnrichedItem;
  onClick?: (item: EnrichedItem) => void;
  size?: "sm" | "md";
}

export function ProductCard({ item, onClick, size = "md" }: ProductCardProps) {
  const imgH = size === "sm" ? "h-36" : "h-64";
  const titleLines = size === "sm" ? "line-clamp-1 text-xs" : "line-clamp-2 text-sm";

  // Clean up title — strip "(Category)" suffix that Snitch adds
  const title = item.title.replace(/\s*\([^)]+\)\s*$/, "").trim();

  return (
    <div
      onClick={() => onClick?.(item)}
      className={`group flex flex-col bg-white rounded-xl overflow-hidden border border-zinc-100 shadow-sm hover:shadow-md transition-all duration-200 ${onClick ? "cursor-pointer hover:border-zinc-300" : ""}`}
    >
      {/* Image */}
      <div className={`relative w-full ${imgH} bg-zinc-50 overflow-hidden`}>
        {item.image_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={item.image_url}
            alt={title}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-zinc-300 text-3xl">
            👗
          </div>
        )}
        {item.slot && (
          <span className="absolute top-2 left-2 bg-black/70 text-white text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-wide">
            {item.slot}
          </span>
        )}
      </div>

      {/* Info */}
      <div className="p-3 flex flex-col gap-1">
        <p className={`font-medium text-zinc-800 ${titleLines}`}>{title}</p>
        <p className="text-xs text-zinc-400">{item.category}</p>
        <p className="text-sm font-semibold text-zinc-900 mt-auto">
          ₹{item.price_inr.toLocaleString("en-IN")}
        </p>
      </div>

      {/* CTA */}
      {item.pdp_url && size === "md" && (
        <a
          href={item.pdp_url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="mx-3 mb-3 text-center text-xs font-semibold text-zinc-600 border border-zinc-200 rounded-lg py-1.5 hover:bg-zinc-900 hover:text-white hover:border-zinc-900 transition-colors"
        >
          Shop Now
        </a>
      )}
    </div>
  );
}
