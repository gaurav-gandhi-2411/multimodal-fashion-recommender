"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, Camera, Zap, Star, Layers } from "lucide-react";
import type { Brand, EnrichedItem } from "@/lib/types";
import { BRAND_META } from "@/lib/types";
import { ProductCard } from "@/components/ProductCard";
import { ProductDrawer } from "@/components/ProductDrawer";
import { Spinner } from "@/components/Spinner";

const BRANDS: Brand[] = ["snitch", "fashor", "powerlook"];

export default function HomePage() {
  const [brand, setBrand] = useState<Brand>("snitch");
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [results, setResults] = useState<EnrichedItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<EnrichedItem | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Brand is passed explicitly so there's no stale-closure risk when the brand tab
  // changes before the previous useCallback re-memoises.
  const runSearch = useCallback(
    async (file: File, activeBrand: Brand) => {
      setSearching(true);
      setError(null);
      setResults([]);
      setSelectedItem(null);

      const form = new FormData();
      form.append("image", file);

      try {
        const res = await fetch(`/api/visual-search?brand=${activeBrand}&k=9`, {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error ?? "Search failed");
        setResults(data.results ?? []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong");
      } finally {
        setSearching(false);
      }
    },
    [] // no brand dep — activeBrand is passed as an argument
  );

  const handleFile = useCallback(
    (file: File, activeBrand: Brand) => {
      if (!file.type.startsWith("image/")) return;
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);
      runSearch(file, activeBrand);
    },
    [runSearch]
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
    // Pass b directly — no setTimeout, no stale closure over the old brand value.
    if (uploadedFile) {
      runSearch(uploadedFile, b);
    }
  };

  const brandMeta = BRAND_META[brand];

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
                {brandMeta.tagline} · Upload any fashion photo to find similar styles
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

          {/* Upload area */}
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
                {/* Uploaded image */}
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
                    className="absolute -top-2 -right-2 w-6 h-6 bg-zinc-900 text-white rounded-full text-xs flex items-center justify-center hover:bg-zinc-700 transition-colors shadow"
                    title="Change image"
                  >
                    ↺
                  </button>
                </div>

                {/* Status */}
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
                      <p className="font-semibold text-zinc-900">
                        {results.length} matches found
                      </p>
                      <p className="text-sm text-zinc-500 mt-1">
                        Sorted by visual similarity · Click any item to see similar styles and outfit ideas
                      </p>
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
        </div>

        {/* Results grid */}
        {results.length > 0 && !searching && (
          <section>
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-zinc-900 text-sm">
                Visual Search Results
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

        {/* Empty state before upload */}
        {!preview && !searching && (
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
