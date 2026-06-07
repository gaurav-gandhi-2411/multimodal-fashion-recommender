from __future__ import annotations

import html
import logging
import re
import time
import urllib.robotparser
import warnings
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests
from pydantic import ValidationError

from app.ingestion.schema import CatalogRow

_MAX_PAGES = 60
_PAGE_SIZE = 250
_PAUSE_S = 0.5
_UA = "Mozilla/5.0 (compatible; fashion-rec-ingest/1.0)"

logger = logging.getLogger(__name__)


class CatalogSource(Protocol):
    def fetch(self) -> list[CatalogRow]: ...


def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class CsvSource:
    """Load a catalog from a CSV file and validate each row."""

    REQUIRED_COLUMNS: frozenset[str] = frozenset(
        {"product_id", "title", "description", "image_url", "price_inr", "category", "pdp_url"}
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch(self) -> list[CatalogRow]:
        df = pd.read_csv(self.path, dtype=str)

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing_cols)}.\n"
                f"Required columns: {sorted(self.REQUIRED_COLUMNS)}\n"
                f"Found columns: {sorted(df.columns.tolist())}"
            )

        valid_rows: list[CatalogRow] = []
        invalid_rows: list[tuple[int, str]] = []

        for raw_idx, row in df.iterrows():
            line_num = int(raw_idx) + 2  # +1 for 0-index, +1 for header row
            try:
                catalog_row = CatalogRow(
                    product_id=str(row["product_id"]),
                    title=str(row["title"]),
                    description=str(row["description"]),
                    image_url=str(row["image_url"]),
                    price_inr=float(row["price_inr"]),
                    category=str(row["category"]),
                    pdp_url=str(row["pdp_url"]),
                )
                valid_rows.append(catalog_row)
            except (ValidationError, ValueError) as exc:
                invalid_rows.append((line_num, str(exc)))

        if invalid_rows:
            lines = "\n".join(f"  Line {ln}: {err}" for ln, err in invalid_rows)
            msg = (
                f"Skipped {len(invalid_rows)} invalid row(s) in {self.path}:\n{lines}\n"
                "Fix these rows or remove them before re-running ingestion."
            )
            if len(invalid_rows) > len(df) // 2:
                raise ValueError(
                    f"Too many invalid rows ({len(invalid_rows)} of {len(df)}) — "
                    "the CSV is likely in the wrong format.\n" + msg
                )
            warnings.warn(msg, stacklevel=2)

        return valid_rows


class ShopifySource:
    """Load a catalog from a Shopify store's public /products.json endpoint."""

    def __init__(self, url: str, *, respect_robots: bool = False) -> None:
        from urllib.parse import urlparse

        parsed = urlparse(url.rstrip("/"))
        self.domain: str = parsed.netloc
        self.base_url: str = f"{parsed.scheme}://{parsed.netloc}"
        self.respect_robots = respect_robots

    def fetch(self) -> list[CatalogRow]:
        session = requests.Session()
        session.headers["User-Agent"] = _UA

        self._check_robots(session)

        probe_url = f"{self.base_url}/products.json?limit=1"
        try:
            probe = session.get(probe_url, timeout=10)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach {self.domain}: {exc}\n"
                "If the store's /products.json endpoint is disabled, "
                "use --source csv instead."
            ) from exc

        if probe.status_code == 404:
            raise RuntimeError(
                f"{self.domain}/products.json returned 404 — "
                "the store has disabled this endpoint.\n"
                "Use --source csv instead: ask the brand to export their catalog."
            )
        if probe.status_code != 200:
            raise RuntimeError(
                f"{self.domain}/products.json returned HTTP {probe.status_code}.\n"
                "Use --source csv instead."
            )

        try:
            probe_json = probe.json()
            if "products" not in probe_json:
                raise ValueError("missing 'products' key")
        except Exception as exc:
            raise RuntimeError(
                f"{self.domain}/products.json does not return valid Shopify JSON: {exc}\n"
                "Use --source csv instead."
            ) from exc

        raw_products = self._paginate(session)
        return self._normalize(raw_products)

    def _check_robots(self, session: requests.Session) -> None:
        rp = urllib.robotparser.RobotFileParser()
        try:
            resp = session.get(f"{self.base_url}/robots.txt", timeout=8)
            rp.parse(resp.text.splitlines())
        except Exception:
            return  # robots.txt unreachable — assume allowed

        allowed = rp.can_fetch(_UA, f"{self.base_url}/products.json")
        if not allowed:
            if self.respect_robots:
                raise RuntimeError(
                    f"{self.domain}/robots.txt disallows /products.json. "
                    "Stopping because --respect-robots is set.\n"
                    "Contact the brand for an alternative data export."
                )
            logger.warning(
                "robots.txt disallows /products.json for %s — proceeding because this is "
                "an authorized client onboarding; pass --respect-robots to enforce.",
                self.domain,
            )

    def _paginate(self, session: requests.Session) -> list[dict]:
        products: list[dict] = []
        seen_ids: set[int] = set()

        for page in range(1, _MAX_PAGES + 1):
            url = f"{self.base_url}/products.json?limit={_PAGE_SIZE}&page={page}"
            try:
                resp = session.get(url, timeout=15)
            except requests.RequestException as exc:
                logger.warning("Shopify page %d failed: %s", page, exc)
                break

            if resp.status_code != 200:
                logger.warning("Shopify page %d returned HTTP %d", page, resp.status_code)
                break

            try:
                batch: list[dict] = resp.json().get("products", [])
            except Exception:
                break

            if not batch:
                break

            first_id = batch[0].get("id")
            if first_id in seen_ids:
                logger.info("Shopify pagination exhausted at page %d (duplicate first ID)", page)
                break

            for p in batch:
                seen_ids.add(p.get("id"))
            products.extend(batch)
            logger.info("Shopify page %d: +%d products (total %d)", page, len(batch), len(products))

            if len(batch) < _PAGE_SIZE:
                break

            time.sleep(_PAUSE_S)

        return products

    def _normalize(self, products: list[dict]) -> list[CatalogRow]:
        rows: list[CatalogRow] = []
        skipped = 0

        for p in products:
            try:
                variants = p.get("variants") or []
                price_str = variants[0].get("price", "0") if variants else "0"
                price_inr = float(price_str)

                images = p.get("images") or []
                image_url = str(images[0].get("src", "")) if images else ""

                raw_description = _strip_html(p.get("body_html") or "")
                product_type = str(p.get("product_type") or "").strip() or "fashion"
                description = raw_description if raw_description else product_type

                handle = str(p.get("handle") or "")
                pdp_url = f"{self.base_url}/products/{handle}" if handle else ""

                row = CatalogRow(
                    product_id=str(p.get("id", "")),
                    title=str(p.get("title", "")).strip(),
                    description=description,
                    image_url=image_url,
                    price_inr=price_inr,
                    category=product_type,
                    pdp_url=pdp_url,
                )
                rows.append(row)
            except (ValidationError, ValueError):
                skipped += 1

        if skipped:
            logger.info("Shopify normalization: skipped %d products with invalid data", skipped)

        return rows
