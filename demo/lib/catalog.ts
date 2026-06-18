import type { CatalogEntry } from "./types";

// Static require() so bundler includes these at build time.
// fs.readFileSync with a dynamic path is not reliably traceable in Vercel serverless.
/* eslint-disable @typescript-eslint/no-require-imports */
const CATALOGS: Record<string, Record<string, CatalogEntry>> = {
  snitch: require("../public/catalog/snitch.json"),
  fashor: require("../public/catalog/fashor.json"),
  powerlook: require("../public/catalog/powerlook.json"),
};
/* eslint-enable @typescript-eslint/no-require-imports */

export function lookupItem(brand: string, itemId: string): CatalogEntry | null {
  return CATALOGS[brand]?.[itemId] ?? null;
}
