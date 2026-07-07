"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, Camera, Zap, Star, Layers, Search } from "lucide-react";
import type { Brand, EnrichedItem } from "@/lib/types";
import { BRAND_META } from "@/lib/types";
import { ProductCard } from "@/components/ProductCard";
import { ProductDrawer } from "@/components/ProductDrawer";
import { Spinner } from "@/components/Spinner";

type SearchMode = "image" | "text";

const BRANDS: Brand[] = ["snitch", "fashor", "powerlook"];

// IMAGE search only — match_confidence is top-1 minus min(top-k) CLIP *image-image*
// cosine score. Calibrated 2026-07-07 against the live index across all 3 brands
// (45 real catalog self-matches + cross-brand/OOD queries):
//   genuine self-match floor:  0.0372 (snitch/fashor; powerlook floor 0.0543)
//   OOD / cross-brand ceiling: 0.030  (scripts/calibrate_match_confidence.py: 0.014-0.030)
// 0.033 sits in that gap. The previous 0.04 clipped ~4/45 genuine self-matches as "low
// confidence" (e.g. snitch aid 571/1008/833 at 0.037-0.038, fashor aid 3011 at 0.0385).
//
// NOT used for style-search (text): audited 24 genuine catalog-title queries + 9
// deliberately-mismatched queries across all 3 brands and the two distributions
// overlap completely (genuine max 0.031, mismatched max 0.031, mismatched queries
// scored HIGHER than genuine ones in 2/3 brands). Text-mode CLIP text-image cosine is
// not on the same scale as image-image cosine and this signal does not separate good
// from bad text queries at all — do not gate or label text results on it until a
// text-specific calibration exists.
const IMAGE_LOW_CONFIDENCE_THRESHOLD = 0.033;

// Extract average color from an image as a 6-digit hex string.
// Uses a 16×16 canvas sample — fast and native, no server round-trip needed.
function extractDominantColor(file: File): Promise<string> {
  return new Promise((resolve) => {
    const canvas = document.createElement("canvas");
    canvas.width = 16;
    canvas.height = 16;
    const ctx = canvas.getContext("2d");
    if (!ctx) { resolve(""); return; }
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, 16, 16);
      const { data } = ctx.getImageData(0, 0, 16, 16);
      let r = 0, g = 0, b = 0;
      const n = 256; // 16×16
      for (let i = 0; i < data.length; i += 4) { r += data[i]; g += data[i + 1]; b += data[i + 2]; }
      const hex = [r, g, b]
        .map((c) => Math.round(c / n).toString(16).padStart(2, "0"))
        .join("");
      URL.revokeObjectURL(img.src);
      resolve(hex);
    };
    img.onerror = () => { URL.revokeObjectURL(img.src); resolve(""); };
    img.src = URL.createObjectURL(file);
  });
}

export default function HomePage() {
  const [brand, setBrand] = useState<Brand>("snitch");
  const [searchMode, setSearchMode] = useState<SearchMode>("image");

  // Image search state
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Text search state
  const [textQuery, setTextQuery] = useState("");
  const [matchConfidence, setMatchConfidence] = useState<number | null>(null);

  // Shared result state
  const [results, setResults] = useState<EnrichedItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<EnrichedItem | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Brand is passed explicitly so there's no stale-closure risk when the brand tab
  // changes before the previous useCallback re-memoises.
  const runImageSearch = useCallback(
    async (file: File, activeBrand: Brand) => {
      setSearching(true);
      setError(null);
      setResults([]);
      setSelectedItem(null);
      setMatchConfidence(null);

      const colorHex = await extractDominantColor(file);
      const form = new FormData();
      form.append("image", file);

      try {
        const colorParam = colorHex ? `&color=${colorHex}` : "";
        const res = await fetch(`/api/visual-search?brand=${activeBrand}&k=9${colorParam}`, {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error ?? "Search failed");
        setResults(data.results ?? []);
        setMatchConfidence(data.match_confidence ?? null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong");
      } finally {
        setSearching(false);
      }
    },
    []
  );

  const runTextSearch = useCallback(
    async (query: string, activeBrand: Brand) => {
      if (!query.trim()) return;
      setSearching(true);
      setError(null);
      setResults([]);
      setSelectedItem(null);
      setMatchConfidence(null);

      try {
        const res = await fetch(
          `/api/style-search?brand=${activeBrand}&k=9&text=${encodeURIComponent(query)}`,
          { method: "POST" }
        );
        const data = await res.json();
        if (!res.ok) throw new Error(data.error ?? "Search failed");
        setResults(data.results ?? []);
        setMatchConfidence(data.match_confidence ?? null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong");
      } finally {
        setSearching(false);
      }
    },
    []
  );

  const handleFile = useCallback(
    (file: File, activeBrand: Brand) => {
      if (!file.type.startsWith("image/")) return;
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);
      runImageSearch(file, activeBrand);
    },
    [runImageSearch]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file, brand);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, brand);
  };

  const onBrandChange = (b: Brand) => {
    setBrand(b);
    setSelectedItem(null);
    if (searchMode === "image" && uploadedFile) {
      runImageSearch(uploadedFile, b);
    } else if (searchMode === "text" && textQuery.trim()) {
      runTextSearch(textQuery, b);
    }
  };

  const onModeChange = (mode: SearchMode) => {
    setSearchMode(mode);
    setResults([]);
    setError(null);
    setMatchConfidence(null);
    setSelectedItem(null);
  };

  const brandMeta = BRAND_META[brand];
  // Only image search has a validated confidence signal (see calibration note above) —
  // text/style search never gates or labels results on matchConfidence.
  const isLowConfidence =
    searchMode === "image" && matchConfidence !== null && matchConfidence < IMAGE_LOW_CONFIDENCE_THRESHOLD;

  return (
    <div className="min-h-screen bg-zinc-50">
      {/* Nav */}
      <nav className="bg-white border-b border-zinc-200 sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="text-zinc-900" size={20} strokeWidth={2.5} />
            <span className="font-bold text-zinc-900 tracking-tight">FashionRec AI</span>
            <span className="text-xs text-zinc-400 font-normal hidden sm:inline">
              Multimodal · India
            </span>
          </div>

          {/* Brand tabs */}
          <div className="flex gap-1 bg-zinc-100 rounded-xl p-1">
            {BRANDS.map((b) => (
              <button
                key={b}
                onClick={() => onBrandChange(b)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  brand === b
                    ? "bg-white text-zinc-900 shadow-sm"
                    : "text-zinc-500 hover:text-zinc-800"
                }`}
              >
                {BRAND_META[b].label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Hero section */}
        <div className="mb-10">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold text-zinc-900">
                Find Your Style
              </h1>
              <p className="text-sm text-zinc-500 mt-1">
                Search by photo or describe what you&apos;re looking for
              </p>
            </div>
            <div className="flex gap-4 text-xs text-zinc-400 hidden md:flex">
              <span className="flex items-center gap-1">
                <Camera size={13} /> Visual Search
              </span>
              <span className="flex items-center gap-1">
                <Star size={13} /> Similar Items
              </span>
              <span className="flex items-center gap-1">
                <Layers size={13} /> Outfit Builder
              </span>
            </div>
          </div>

          {/* Catalog scope — always visible so a shopper doesn't upload e.g. a men's tee
              to a women's-only catalog expecting menswear back. */}
          <div className="flex items-center gap-1.5 mb-4 px-3 py-1.5 bg-zinc-100 rounded-lg w-fit text-xs text-zinc-600">
            <span className="font-semibold text-zinc-800">{brandMeta.label} carries:</span>
            {brandMeta.tagline}
          </div>

          {/* Search mode toggle */}
          <div className="flex gap-1 bg-zinc-100 rounded-xl p-1 w-fit mb-4">
            <button
              onClick={() => onModeChange("image")}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                searchMode === "image"
                  ? "bg-white text-zinc-900 shadow-sm"
                  : "text-zinc-500 hover:text-zinc-800"
              }`}
            >
              <Camera size={14} /> Image Search
            </button>
            <button
              onClick={() => onModeChange("text")}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                searchMode === "text"
                  ? "bg-white text-zinc-900 shadow-sm"
                  : "text-zinc-500 hover:text-zinc-800"
              }`}
            >
              <Search size={14} /> Style Search
            </button>
          </div>

          {searchMode === "text" ? (
            /* Text-search box */
            <div className="bg-white rounded-2xl border border-zinc-200 p-5">
              <form
                onSubmit={(e) => { e.preventDefault(); runTextSearch(textQuery, brand); }}
                className="flex gap-3"
              >
                <input
                  type="text"
                  value={textQuery}
                  onChange={(e) => setTextQuery(e.target.value)}
                  placeholder="e.g. white oversized linen shirt, floral print kurta…"
                  className="flex-1 rounded-xl border border-zinc-200 px-4 py-3 text-sm text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 focus:border-transparent"
                />
                <button
                  type="submit"
                  disabled={!textQuery.trim() || searching}
                  className="px-5 py-3 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                  {searching ? <Spinner size={16} /> : <Search size={16} />}
                  Search
                </button>
              </form>

              {/* Status indicator — no confidence verdict here: audited 2026-07-07 and
                  match_confidence does not separate good from bad text queries (see
                  calibration note above). Showing a "Low match" claim here would be
                  actively misleading, not just uninformative. */}
              <div className="mt-3 min-h-5">
                {searching ? (
                  <p className="text-xs text-zinc-400">Searching {brandMeta.label} catalog…</p>
                ) : error ? (
                  <p className="text-xs text-red-500">{error}</p>
                ) : null}
              </div>
            </div>
          ) : (
            /* Image upload area */
            <>
              <div
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer overflow-hidden ${
                  dragging
                    ? "border-zinc-900 bg-zinc-100"
                    : preview
                    ? "border-transparent"
                    : "border-zinc-300 hover:border-zinc-400 bg-white"
                }`}
                style={{ minHeight: preview ? undefined : "200px" }}
              >
                {preview ? (
                  <div className="flex items-start gap-6 p-4">
                    <div className="flex-shrink-0 relative">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={preview}
                        alt="Uploaded"
                        className="w-32 h-40 object-cover rounded-xl border border-zinc-200 shadow-sm"
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          fileInputRef.current?.click();
                        }}
                        className="absolute -top-2 -right-2 h-6 bg-zinc-900 text-white rounded-full text-[10px] font-semibold flex items-center gap-1 px-2 hover:bg-zinc-700 transition-colors shadow"
                        title="Change image"
                      >
                        ↺<span className="hidden sm:inline">Change</span>
                      </button>
                    </div>
                    <div className="flex-1 flex flex-col justify-center py-4">
                      {searching ? (
                        <div className="flex items-center gap-3 text-zinc-600">
                          <Spinner size={20} />
                          <span className="text-sm font-medium">
                            Searching {brandMeta.label} catalog…
                          </span>
                        </div>
                      ) : results.length > 0 ? (
                        <div>
                          {isLowConfidence ? (
                            <p className="font-semibold text-zinc-900">No strong match found</p>
                          ) : (
                            <>
                              <p className="font-semibold text-zinc-900">
                                {results.length} matches found
                              </p>
                              <p className="text-sm text-zinc-500 mt-1">
                                Sorted by visual similarity · Click any item to see similar styles and outfit ideas
                              </p>
                            </>
                          )}
                          {matchConfidence !== null && (
                            <p className={`text-xs font-medium mt-2 ${isLowConfidence ? "text-amber-500" : "text-emerald-600"}`}>
                              {isLowConfidence
                                ? `Low catalog match (confidence ${matchConfidence.toFixed(3)}) — ${brandMeta.label} catalog may lack this style`
                                : `Strong catalog match (confidence ${matchConfidence.toFixed(3)})`}
                            </p>
                          )}
                        </div>
                      ) : error ? (
                        <p className="text-sm text-red-600">{error}</p>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center gap-3 p-10 text-center">
                    <div className="w-14 h-14 rounded-2xl bg-zinc-100 flex items-center justify-center">
                      <Upload className="text-zinc-500" size={24} />
                    </div>
                    <div>
                      <p className="font-semibold text-zinc-700">Drop a photo here</p>
                      <p className="text-sm text-zinc-400 mt-1">
                        Or click to browse · JPG, PNG, WEBP
                      </p>
                    </div>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onInputChange}
              />
            </>
          )}
        </div>

        {/* Skeleton while searching */}
        {searching && (searchMode === "text" || preview) && (
          <section>
            <div className="flex items-center gap-3 mb-4 h-5">
              <div className="h-3 bg-zinc-200 rounded-full w-40 animate-pulse" />
              <div className="h-3 bg-zinc-200 rounded-full w-16 animate-pulse" />
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {[...Array(9)].map((_, i) => (
                <div key={i} className="rounded-xl bg-white border border-zinc-100 overflow-hidden">
                  <div className="h-64 bg-zinc-100 animate-pulse" />
                  <div className="p-3 space-y-2">
                    <div className="h-2.5 bg-zinc-100 rounded-full w-3/4 animate-pulse" />
                    <div className="h-2.5 bg-zinc-100 rounded-full w-1/4 animate-pulse" />
                    <div className="h-8 bg-zinc-100 rounded-lg w-full animate-pulse mt-1" />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Low-confidence empty state (image search only — see calibration note above).
            A low match_confidence means no catalog item stands out from the pool;
            showing those marginal results reads as a broken model. Treat it as a
            deliberate, labeled outcome instead. */}
        {results.length > 0 && !searching && isLowConfidence && (
          <div className="mt-4 py-12 text-center">
            <p className="text-sm font-semibold text-zinc-700">
              No strong matches in {brandMeta.label}&apos;s catalog
            </p>
            <p className="text-xs text-zinc-400 mt-1 max-w-md mx-auto">
              {brandMeta.label} carries {brandMeta.tagline}. This photo doesn&apos;t closely match
              anything in stock — try a different brand tab, or describe what you&apos;re looking for instead.
            </p>
            <button
              onClick={() => onModeChange("text")}
              className="mt-4 px-3 py-1.5 rounded-lg text-xs font-medium bg-zinc-900 text-white hover:bg-zinc-700 transition-colors"
            >
              Switch to Style Search
            </button>
          </div>
        )}

        {/* Results grid */}
        {results.length > 0 && !searching && !isLowConfidence && (
          <section>
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-zinc-900 text-sm">
                {searchMode === "text" ? "Style Search Results" : "Visual Search Results"}
                <span className="ml-2 text-zinc-400 font-normal">{brandMeta.label}</span>
              </h2>
              <p className="text-xs text-zinc-400">
                Click a product to explore similar items &amp; outfit ideas
              </p>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {results.map((item) => (
                <ProductCard
                  key={item.item_id}
                  item={item}
                  onClick={setSelectedItem}
                />
              ))}
            </div>
          </section>
        )}

        {/* Zero-results state */}
        {!searching && (searchMode === "text" || preview) && results.length === 0 && !error && (
          searchMode === "text" && !textQuery.trim() ? null : (
            <div className="mt-4 py-12 text-center text-zinc-400">
              <p className="text-sm font-medium">No matches found</p>
              <p className="text-xs mt-1">
                {searchMode === "text"
                  ? "Try rephrasing your query or check if this brand carries this style."
                  : "Try a clearer fashion photo or one with better lighting."}
              </p>
            </div>
          )
        )}

        {/* Empty state before search */}
        {!preview && !searching && searchMode === "image" && (
          <div className="mt-4 grid grid-cols-3 gap-4 opacity-60 pointer-events-none select-none">
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="rounded-xl bg-white border border-zinc-100 overflow-hidden"
              >
                <div className="h-52 bg-gradient-to-br from-zinc-100 to-zinc-50" />
                <div className="p-3 space-y-2">
                  <div className="h-3 bg-zinc-100 rounded-full w-3/4" />
                  <div className="h-3 bg-zinc-100 rounded-full w-1/4" />
                  <div className="h-3 bg-zinc-100 rounded-full w-1/3" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Feature pills */}
        <div className="mt-16 flex flex-wrap gap-3 justify-center">
          {[
            { icon: "🔍", label: "CLIP ViT-B/32 Visual Search" },
            { icon: "💬", label: "Natural-Language Style Search" },
            { icon: "✨", label: "MMR Diversity Reranking" },
            { icon: "👔", label: "Outfit Slot Completion" },
            { icon: "⚡", label: "Groq LLM Explanations" },
            { icon: "🏷️", label: "Category + Price Affinity" },
          ].map((f) => (
            <span
              key={f.label}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-zinc-200 rounded-full text-xs text-zinc-600"
            >
              <span>{f.icon}</span>
              {f.label}
            </span>
          ))}
        </div>

        <p className="text-center text-xs text-zinc-400 mt-6">
          Powered by a multimodal two-tower model trained on H&amp;M (913k interactions) ·
          3.06× lift over popularity baseline · 2.12× over co-purchase CF
        </p>
      </main>

      {/* Drawer */}
      <ProductDrawer
        item={selectedItem}
        brand={brand}
        onClose={() => setSelectedItem(null)}
      />
    </div>
  );
}
