import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "cdn.shopify.com" },
      { protocol: "https", hostname: "**.shopify.com" },
      { protocol: "https", hostname: "fashor.com" },
      { protocol: "https", hostname: "**.fashor.com" },
      { protocol: "https", hostname: "powerlook.in" },
      { protocol: "https", hostname: "snitch.co.in" },
      { protocol: "https", hostname: "snitch.com" },
      { protocol: "https", hostname: "www.snitch.com" },
      { protocol: "https", hostname: "**.cloudfront.net" },
      { protocol: "https", hostname: "images.unsplash.com" },
    ],
  },
};

export default nextConfig;
