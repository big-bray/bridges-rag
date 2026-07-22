"""Orchestrates scraping a Bridges proceedings year end to end."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from bridges_rag.ingest.bibtex import merge_bibtex, parse_bibtex
from bridges_rag.ingest.client import Scraper
from bridges_rag.ingest.detail import merge_detail, parse_detail
from bridges_rag.ingest.download import download_pdf, harvest_pdf
from bridges_rag.ingest.listing import parse_listing
from bridges_rag.ingest.manifest import read_manifest, write_manifest
from bridges_rag.ingest.models import Paper

logger = logging.getLogger(__name__)


def archive_url(year: int) -> str:
    return f"https://archive.bridgesmathart.org/{year}/"


def ingest_year(
    year: int,
    data_dir: Path,
    *,
    limit: int | None = None,
    delay: float = 1.0,
    persist_pdf: bool = False,
) -> list[Paper]:
    base_url = archive_url(year)
    manifest_path = data_dir / str(year) / "manifest.jsonl"
    previous = (
        {p.paper_id: p for p in read_manifest(manifest_path)} if manifest_path.exists() else {}
    )

    with Scraper(delay=delay) as scraper:
        listing_html = scraper.get(base_url).text
        entries = parse_listing(listing_html, year=year, base_url=base_url)
        if limit is not None:
            entries = entries[:limit]

        papers: list[Paper] = []
        for entry in entries:
            paper = entry
            if paper.detail_url is not None:
                try:
                    detail_html = scraper.get(paper.detail_url).text
                    paper = merge_detail(paper, parse_detail(detail_html))
                except httpx.HTTPError:
                    logger.exception("failed to fetch detail page for %s", paper.paper_id)

            if paper.bibtex_url is not None:
                try:
                    bibtex_text = scraper.get(paper.bibtex_url).text
                    paper = merge_bibtex(paper, bibtex_text, parse_bibtex(bibtex_text))
                except httpx.HTTPError:
                    logger.exception("failed to fetch bibtex for %s", paper.paper_id)

            try:
                if persist_pdf:
                    paper = download_pdf(paper, data_dir, scraper)
                else:
                    paper = harvest_pdf(
                        paper, data_dir, scraper, previous=previous.get(paper.paper_id)
                    )
            except httpx.HTTPError:
                logger.exception("failed to download PDF for %s", paper.paper_id)

            papers.append(paper)

    write_manifest(papers, manifest_path)
    return papers
