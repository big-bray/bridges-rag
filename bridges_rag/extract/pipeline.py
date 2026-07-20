"""Orchestrates markdown extraction for a downloaded Bridges proceedings year."""

from __future__ import annotations

import logging
from pathlib import Path

from bridges_rag.extract.models import ExtractedPaper
from bridges_rag.extract.pdf import extract_pages
from bridges_rag.ingest.manifest import read_manifest
from bridges_rag.ingest.models import Paper

logger = logging.getLogger(__name__)


def extract_paper(paper: Paper, data_dir: Path, markdown_dir: Path) -> ExtractedPaper:
    """Extract one paper's downloaded PDF to a markdown file in `markdown_dir`.

    Never raises: PyMuPDF can throw a variety of internal errors on
    malformed or unusual PDFs, so failures are caught, logged, and recorded
    on the returned `ExtractedPaper.error` instead of dying, keeping one bad
    paper from stopping the rest of the batch.
    """
    if paper.pdf_path is None:
        return ExtractedPaper(paper_id=paper.paper_id, pages=[], error="no downloaded PDF")

    pdf_path = data_dir / paper.pdf_path
    try:
        pages = extract_pages(pdf_path, first_page=paper.first_page)
    except Exception as exc:
        logger.exception("failed to extract markdown for %s", paper.paper_id)
        return ExtractedPaper(paper_id=paper.paper_id, pages=[], error=str(exc))

    extracted = ExtractedPaper(paper_id=paper.paper_id, pages=pages)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / f"{paper.paper_id}.md").write_text(render_markdown(extracted), encoding="utf-8")
    return extracted


def render_markdown(extracted: ExtractedPaper) -> str:
    """Join a paper's pages into one markdown document.

    Each page is preceded by an HTML comment marker recording its PDF and
    proceedings page numbers, so a later chunking step can attribute page
    numbers to chunks without re-running extraction.
    """
    blocks = []
    for page in extracted.pages:
        marker = f"<!-- pdf_page={page.pdf_page} proceedings_page={page.proceedings_page} -->"
        blocks.append(f"{marker}\n\n{page.markdown}".rstrip())
    return "\n\n".join(blocks) + "\n"


def extract_year(year: int, data_dir: Path, *, limit: int | None = None) -> list[ExtractedPaper]:
    """Extract markdown for every downloaded paper in `year`'s manifest."""
    manifest_path = data_dir / str(year) / "manifest.jsonl"
    papers = read_manifest(manifest_path)
    if limit is not None:
        papers = papers[:limit]

    markdown_dir = data_dir / str(year) / "markdown"
    return [extract_paper(paper, data_dir, markdown_dir) for paper in papers]
