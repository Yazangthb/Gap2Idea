"""Semantic Scholar Graph API client.

Replaces the bulky arxiv-metadata-oai-snapshot.json dump with on-demand lookups.

Endpoints used:
- GET  /graph/v1/paper/search          -> keyword/title search
- POST /graph/v1/paper/batch           -> bulk metadata lookup
- GET  /graph/v1/paper/{id}            -> single paper lookup

Auth:
- Set S2_API_KEY in your environment to raise rate limits. Without it, the
  unauthenticated tier is ~100 requests / 5 minutes shared across the IP.
"""
from __future__ import annotations

import os
import time
from typing import Iterable

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.semanticscholar.org/graph/v1"

DEFAULT_FIELDS = (
    "paperId,title,year,venue,abstract,url,authors,"
    "citationCount,openAccessPdf,externalIds"
)


class S2Client:
    """Thin wrapper with retry-on-429 + optional API key."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: float = 30.0,
        backoff_base: float = 2.0,
    ):
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})

    # ---- low-level ----
    def _request(self, method: str, url: str, **kwargs) -> dict:
        for attempt in range(self.max_retries):
            r = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if r.status_code == 429:
                # exponential backoff: 1s, 2s, 4s, 8s, 16s
                sleep_s = self.backoff_base ** attempt
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.json()
        # final attempt without swallowing the error
        r = self.session.request(method, url, timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r.json()

    # ---- high-level ----
    def search(self, query: str, limit: int = 10, fields: str = DEFAULT_FIELDS) -> list[dict]:
        """Keyword/title search. Returns list of paper records."""
        data = self._request(
            "GET",
            f"{BASE}/paper/search",
            params={"query": query, "limit": limit, "fields": fields},
        )
        return data.get("data", [])

    def get_by_arxiv(self, arxiv_id: str, fields: str = DEFAULT_FIELDS) -> dict | None:
        """Look up a single paper by its arXiv ID (e.g. '1912.08927')."""
        try:
            return self._request(
                "GET",
                f"{BASE}/paper/ARXIV:{arxiv_id}",
                params={"fields": fields},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def get_batch_by_arxiv(
        self,
        arxiv_ids: Iterable[str],
        fields: str = DEFAULT_FIELDS,
        chunk_size: int = 100,
    ) -> list[dict | None]:
        """Bulk lookup by arXiv ID. Preserves input order; missing papers => None.

        S2 caps /paper/batch at 500 IDs per call; we use 100 for safer pacing.
        """
        ids = [f"ARXIV:{a}" for a in arxiv_ids]
        results: list[dict | None] = []
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            data = self._request(
                "POST",
                f"{BASE}/paper/batch",
                params={"fields": fields},
                json={"ids": chunk},
            )
            # The batch endpoint returns a list of papers (or null) in input order.
            results.extend(data)
        return results


def flatten_paper(p: dict | None, fallback_id: str = "") -> dict:
    """Coerce an S2 paper record into a flat row suitable for a TSV.

    The `id` column is always the lookup key (`fallback_id`) so it joins
    cleanly with `gaps_with_clusters.tsv`. S2's reported arXiv ID is kept
    separately as `s2_arxiv_id` in case S2 returned a different paper than
    we asked for (this happens occasionally for ambiguous IDs).
    """
    if not p:
        return {
            "id": fallback_id,
            "s2_arxiv_id": "",
            "paperId": "",
            "title": "",
            "abstract": "",
            "year": "",
            "venue": "",
            "citation_count": "",
            "openaccess_pdf": "",
            "url": "",
        }
    oa = p.get("openAccessPdf") or {}
    ext = p.get("externalIds") or {}
    return {
        "id": fallback_id,
        "s2_arxiv_id": ext.get("ArXiv") or "",
        "paperId": p.get("paperId") or "",
        "title": p.get("title") or "",
        "abstract": p.get("abstract") or "",
        "year": p.get("year") or "",
        "venue": p.get("venue") or "",
        "citation_count": p.get("citationCount") or 0,
        "openaccess_pdf": oa.get("url") or "",
        "url": p.get("url") or "",
    }
