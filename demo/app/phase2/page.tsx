"use client";

import { useState } from "react";
import { Zap, Users, TrendingUp, Info, ChevronRight } from "lucide-react";
import phase2Data from "../../public/phase2_users.json";

// ── Types ─────────────────────────────────────────────────────────────────────

interface HMItem {
  article_id: string;
  prod_name: string;
  product_type_name: string;
  product_group_name?: string;
  colour_group_name: string;
  department_name: string;
  score?: number;
}

interface DemoUser {
  user_id: string;
  display_id: string;
  segment_id: number;
  segment_label: string;
  segment_top_types: string[];
  user_top_types: string[];
  train_purchase_count: number;
  test_purchase_count: number;
  history_sample: HMItem[];
  recs: HMItem[];
  request_id: string;
}

interface Segment {
  id: number;
  label: string;
  top_types: string[];
}

// ── Colour swatch map ─────────────────────────────────────────────────────────

const COLOUR_MAP: Record<string, string> = {
  Black: "#1c1c1c",
  White: "#f5f5f5",
  "Off White": "#f0ede4",
  Blue: "#2563eb",
  "Dark Blue": "#1e3a5f",
  "Light Blue": "#93c5fd",
  "Other Blue": "#3b82f6",
  Red: "#dc2626",
  "Dark Red": "#991b1b",
  "Light Red": "#fca5a5",
  "Other Red": "#ef4444",
  Pink: "#ec4899",
  "Dark Pink": "#be185d",
  "Light Pink": "#fbcfe8",
  "Other Pink": "#f472b6",
  Green: "#16a34a",
  "Dark Green": "#14532d",
  "Light Green": "#86efac",
  "Greenish Khaki": "#6b7c3c",
  "Other Green": "#22c55e",
  Grey: "#6b7280",
  "Dark Grey": "#374151",
  "Light Grey": "#d1d5db",
  Beige: "#d4b896",
  "Dark Beige": "#b8956a",
  "Light Beige": "#f5e6d3",
  "Greyish Beige": "#c4b8a8",
  Yellow: "#eab308",
  "Dark Yellow": "#a16207",
  "Light Yellow": "#fef08a",
  "Other Yellow": "#facc15",
  "Yellowish Brown": "#92683a",
  Orange: "#f97316",
  "Dark Orange": "#c2410c",
  "Light Orange": "#fdba74",
  "Other Orange": "#fb923c",
  Purple: "#7c3aed",
  "Dark Purple": "#581c87",
  "Light Purple": "#c4b5fd",
  "Other Purple": "#a78bfa",
  Turquoise: "#0d9488",
  "Dark Turquoise": "#0f766e",
  "Light Turquoise": "#99f6e4",
  "Other Turquoise": "#2dd4bf",
  Brown: "#92400e",
  Gold: "#b45309",
  Silver: "#9ca3af",
  "Bronze/Copper": "#b07030",
  Transparent: "#e5e7eb",
  Unknown: "#9ca3af",
  Other: "#9ca3af",
};

function ColourSwatch({ colour, size = "md" }: { colour: string; size?: "sm" | "md" }) {
  const bg = COLOUR_MAP[colour] ?? "#9ca3af";
  const isLight = ["White", "Off White", "Light Grey", "Light Yellow", "Light Beige", "Transparent"].includes(colour);
  const dim = size === "sm" ? "w-6 h-6" : "w-8 h-8";
  return (
    <div
      className={`${dim} rounded-full flex-shrink-0 border ${isLight ? "border-zinc-300" : "border-transparent"}`}
      style={{ backgroundColor: bg }}
      title={colour}
    />
  );
}

function ItemCard({ item }: { item: HMItem }) {
  const bg = COLOUR_MAP[item.colour_group_name] ?? "#9ca3af";
  const isLight = ["White", "Off White", "Light Grey", "Light Yellow", "Light Beige", "Transparent"].includes(
    item.colour_group_name
  );
  return (
    <div className="flex flex-col items-center gap-1.5 group">
      <div
        className={`w-14 h-18 rounded-lg border ${isLight ? "border-zinc-300" : "border-zinc-100"} shadow-sm transition-transform group-hover:scale-105`}
        style={{ backgroundColor: bg, height: "4.5rem" }}
      />
      <div className="text-center w-14">
        <p className="text-[10px] font-medium text-zinc-700 leading-tight line-clamp-2">{item.prod_name}</p>
        <p className="text-[9px] text-zinc-400 truncate">{item.product_type_name}</p>
      </div>
    </div>
  );
}

function UserCard({ user, isSelected, onClick }: { user: DemoUser; isSelected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-xl border p-3 transition-all duration-150 ${
        isSelected
          ? "border-zinc-900 bg-zinc-900 text-white shadow-md"
          : "border-zinc-200 bg-white hover:border-zinc-400 text-zinc-800"
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <p className={`text-xs font-semibold ${isSelected ? "text-white" : "text-zinc-900"}`}>
            {user.display_id}
          </p>
          <p className={`text-[10px] ${isSelected ? "text-zinc-300" : "text-zinc-500"}`}>
            {user.train_purchase_count} train purchases
          </p>
        </div>
        <ChevronRight size={14} className={isSelected ? "text-zinc-400" : "text-zinc-300"} />
      </div>
      <div className="flex flex-wrap gap-1">
        {user.user_top_types.slice(0, 2).map((t) => (
          <span
            key={t}
            className={`text-[9px] px-1.5 py-0.5 rounded-full font-medium ${
              isSelected ? "bg-zinc-700 text-zinc-200" : "bg-zinc-100 text-zinc-600"
            }`}
          >
            {t}
          </span>
        ))}
      </div>
    </button>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Phase2Page() {
  const { users, segments, trending } = phase2Data as {
    users: DemoUser[];
    segments: Segment[];
    trending: HMItem[];
  };

  const [activeSegment, setActiveSegment] = useState<number>(0);
  const [selectedUser, setSelectedUser] = useState<DemoUser>(
    users.find((u) => u.segment_id === 0) ?? users[0]
  );
  const [showTrending, setShowTrending] = useState(false);

  const segmentUsers = users.filter((u) => u.segment_id === activeSegment);

  const handleSegmentChange = (segId: number) => {
    setActiveSegment(segId);
    const first = users.find((u) => u.segment_id === segId);
    if (first) setSelectedUser(first);
  };

  return (
    <div className="min-h-screen bg-zinc-50">
      {/* Nav */}
      <nav className="bg-white border-b border-zinc-200 sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="text-zinc-900" size={20} strokeWidth={2.5} />
            <span className="font-bold text-zinc-900 tracking-tight">FashionRec AI</span>
            <span className="text-xs text-zinc-400 font-normal hidden sm:inline">
              Personalization Engine · H&amp;M Dataset
            </span>
          </div>
          <a href="/" className="text-xs text-zinc-500 hover:text-zinc-900 transition-colors">
            ← Visual Search Demo
          </a>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-zinc-900">Personalization Engine</h1>
              <p className="text-sm text-zinc-500 mt-1">
                Two-tower model · 913K interactions · H&amp;M Fashion Dataset
              </p>
            </div>
            {/* Metrics */}
            <div className="hidden md:flex gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-zinc-900">3.06×</p>
                <p className="text-[10px] text-zinc-500">vs popularity baseline</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-zinc-900">2.12×</p>
                <p className="text-[10px] text-zinc-500">vs co-purchase CF</p>
              </div>
            </div>
          </div>

          {/* Honest framing callout */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex gap-3">
            <Info size={16} className="text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-blue-800 space-y-1">
              <p>
                <strong>12 held-out test users</strong> — the model was trained on a separate cohort.
                These users&apos; purchase histories come from the <strong>train split only</strong>;
                their test-split behaviour was never seen during training.
              </p>
              <p>
                Recommendations are fetched live from the same{" "}
                <code className="bg-blue-100 px-1 rounded">/v1/h_and_m/recommend</code> endpoint your curl
                hit. In production: swap in your own customer IDs and transaction data — same API, same model
                architecture.
              </p>
            </div>
          </div>
        </div>

        {/* Segment tabs */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Users size={14} className="text-zinc-500" />
            <span className="text-xs font-medium text-zinc-600 uppercase tracking-wide">Style Segments</span>
            <span className="text-[10px] text-zinc-400 ml-1">K-means k=6 on product-type distribution</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {segments.map((seg) => (
              <button
                key={seg.id}
                onClick={() => handleSegmentChange(seg.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                  activeSegment === seg.id
                    ? "bg-zinc-900 text-white shadow-sm"
                    : "bg-white border border-zinc-200 text-zinc-600 hover:border-zinc-400"
                }`}
              >
                {seg.label}
              </button>
            ))}
          </div>
        </div>

        {/* Main content: user selector + recs */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* User selector */}
          <div className="lg:col-span-1">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wide mb-3">
              Demo Users
            </p>
            <div className="space-y-2">
              {segmentUsers.map((user) => (
                <UserCard
                  key={user.user_id}
                  user={user}
                  isSelected={selectedUser.user_id === user.user_id}
                  onClick={() => setSelectedUser(user)}
                />
              ))}
            </div>

            {/* User stats */}
            <div className="mt-4 p-3 bg-white rounded-xl border border-zinc-200">
              <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide mb-2">
                Selected User
              </p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-zinc-500">Train purchases</span>
                  <span className="font-medium text-zinc-900">{selectedUser.train_purchase_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Test purchases</span>
                  <span className="font-medium text-zinc-700">{selectedUser.test_purchase_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Top style</span>
                  <span className="font-medium text-zinc-900 text-right max-w-[100px] truncate">
                    {selectedUser.user_top_types[0] ?? "—"}
                  </span>
                </div>
              </div>
              <div className="mt-2 pt-2 border-t border-zinc-100">
                <p className="text-[10px] text-zinc-400">
                  ID: {selectedUser.user_id.slice(0, 12)}…
                </p>
              </div>
            </div>
          </div>

          {/* Personalized recs */}
          <div className="lg:col-span-3">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wide">
                  Personalized Recommendations
                </p>
                <p className="text-[10px] text-zinc-400 mt-0.5">
                  Live from /v1/h_and_m/recommend · user tower inference at request time
                </p>
              </div>
              <button
                onClick={() => setShowTrending(!showTrending)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                  showTrending
                    ? "bg-zinc-900 text-white border-zinc-900"
                    : "bg-white text-zinc-600 border-zinc-200 hover:border-zinc-400"
                }`}
              >
                <TrendingUp size={12} />
                Compare baseline
              </button>
            </div>

            {!showTrending ? (
              /* Personalized */
              <div className="bg-white rounded-xl border border-zinc-200 p-5">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xs font-semibold text-zinc-900">
                    {selectedUser.display_id} · {selectedUser.segment_label} segment
                  </span>
                  <span className="text-[10px] text-zinc-400">
                    train history: {selectedUser.train_purchase_count} tx
                  </span>
                </div>
                <div className="grid grid-cols-4 sm:grid-cols-8 gap-3">
                  {selectedUser.recs.map((item) => (
                    <ItemCard key={item.article_id} item={item} />
                  ))}
                </div>

                {/* History sample */}
                {selectedUser.history_sample.length > 0 && (
                  <div className="mt-5 pt-4 border-t border-zinc-100">
                    <p className="text-[10px] font-medium text-zinc-400 uppercase tracking-wide mb-2">
                      Recent train-split purchases (fed as history)
                    </p>
                    <div className="flex gap-3">
                      {selectedUser.history_sample.map((item) => (
                        <div key={item.article_id} className="flex items-center gap-1.5">
                          <ColourSwatch colour={item.colour_group_name} size="sm" />
                          <span className="text-[10px] text-zinc-500 max-w-[60px] truncate">
                            {item.prod_name}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <p className="text-[10px] text-zinc-400 mt-4">
                  request_id: {selectedUser.request_id}
                </p>
              </div>
            ) : (
              /* Baseline */
              <div className="bg-white rounded-xl border border-zinc-200 p-5">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp size={14} className="text-zinc-500" />
                  <span className="text-xs font-semibold text-zinc-900">
                    Popularity Baseline — same items for every user
                  </span>
                </div>
                <div className="grid grid-cols-4 sm:grid-cols-8 gap-3">
                  {trending.map((item) => (
                    <ItemCard key={item.article_id} item={item} />
                  ))}
                </div>
                <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <p className="text-xs text-amber-800">
                    <strong>Popularity baseline</strong>: top-8 most purchased items in train split —
                    no personalization, identical for all users. The two-tower model achieves{" "}
                    <strong>3.06× Recall@10</strong> over this baseline on the held-out test set.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer context */}
        <div className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs text-zinc-500">
          <div className="bg-white border border-zinc-200 rounded-xl p-4">
            <p className="font-semibold text-zinc-700 mb-1">Model</p>
            <p>Two-tower (CLIP + SBERT item tower, Transformer user tower) · FAISS IndexFlatIP · 256-dim fused embeddings</p>
          </div>
          <div className="bg-white border border-zinc-200 rounded-xl p-4">
            <p className="font-semibold text-zinc-700 mb-1">Data</p>
            <p>H&amp;M Fashion Dataset · 913K train interactions · 48K customers · 10.5K articles · test split held out</p>
          </div>
          <div className="bg-white border border-zinc-200 rounded-xl p-4">
            <p className="font-semibold text-zinc-700 mb-1">For your brand</p>
            <p>Replace with your transaction data. Same API, same model architecture. Brand-isolated: your data never mixes with others.</p>
          </div>
        </div>

        <p className="text-center text-[10px] text-zinc-400 mt-6">
          Items displayed as colour swatches — H&amp;M Kaggle dataset has no hotlinkable CDN image URLs.
          In a production integration, product images would be fetched from your catalogue.
        </p>
      </main>
    </div>
  );
}
