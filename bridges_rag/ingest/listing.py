"""Parse a Bridges archive year-listing page into `Paper` stubs."""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from bridges_rag.ingest.models import FRONT_MATTER_CATEGORY, Paper

# Bridges uses an en dash ("Pages 29–36"); a plain hyphen is accepted too.
_PAGE_RANGE_RE = re.compile(r"Pages\s+(\d+)(?:\s*[-–—]\s*(\d+))?")


def parse_listing(html: str, year: int, base_url: str) -> list[Paper]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not isinstance(main, Tag):
        raise ValueError("listing page has no <main> content area")

    papers: list[Paper] = []
    category = FRONT_MATTER_CATEGORY
    for el in main.find_all(["div", "a"]):
        if el.name == "div" and "papercategory" in el.get("class", []):
            category = el.get_text(strip=True)
        elif el.name == "a" and "paperentry" in el.get("class", []):
            papers.append(_parse_entry(el, category=category, year=year, base_url=base_url))
    return papers


def _parse_entry(anchor: Tag, *, category: str, year: int, base_url: str) -> Paper:
    href = anchor["href"]
    assert isinstance(href, str)

    title_el = anchor.find("div", class_="papertitle")
    authors_el = anchor.find("div", class_="paperauthors")
    pages_el = anchor.find("div", class_="paperpages")

    title = title_el.get_text(strip=True) if title_el else ""
    authors = _split_authors(authors_el.get_text(strip=True)) if authors_el else []

    first_page = last_page = None
    if pages_el is not None:
        match = _PAGE_RANGE_RE.search(pages_el.get_text(strip=True))
        if match:
            first_page = int(match.group(1))
            last_page = int(match.group(2)) if match.group(2) else first_page

    is_direct_pdf = href.endswith(".pdf")
    paper_id = href.rsplit("/", 1)[-1].removesuffix(".pdf").removesuffix(".html")
    pdf_href = href if is_direct_pdf else f"{href.removesuffix('.html')}.pdf"
    bibtex_href = None if is_direct_pdf else f"{href.removesuffix('.html')}-bibtex.txt"

    return Paper(
        paper_id=paper_id,
        year=year,
        title=title,
        authors=authors,
        category=category,
        first_page=first_page,
        last_page=last_page,
        detail_url=None if is_direct_pdf else urljoin(base_url, href),
        pdf_url=urljoin(base_url, pdf_href),
        bibtex_url=None if bibtex_href is None else urljoin(base_url, bibtex_href),
    )


def _split_authors(text: str) -> list[str]:
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return []

    if len(parts) == 1 and " and " in parts[0]:
        return [name.strip() for name in parts[0].split(" and ")]

    last = parts[-1]
    if last.startswith("and "):
        parts[-1] = last[len("and ") :].strip()
    return parts
