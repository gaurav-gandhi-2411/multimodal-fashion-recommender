import fs from "fs";
import path from "path";
import type { CatalogEntry } from "./types";

const catalogCache: Record<string, Record<string, CatalogEntry>> = {};

export function getCatalog(brand: string): Record<string, CatalogEntry> {
  if (!catalogCache[brand]) {
    const filePath = path.join(process.cwd(), "public", "catalog", `${brand}.json`);
    catalogCache[brand] = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  }
  return catalogCache[brand];
}

export function lookupItem(brand: string, itemId: string): CatalogEntry | null {
  const catalog = getCatalog(brand);
  return catalog[itemId] ?? null;
}
