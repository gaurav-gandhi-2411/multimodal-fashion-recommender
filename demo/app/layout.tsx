import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FashionRec AI — Multimodal Style Discovery",
  description:
    "AI-powered fashion recommendations for Indian D2C brands. Visual search, similar items, and outfit completion.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
