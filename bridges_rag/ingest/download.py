"""PDF downloads."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from bridges_rag.extract.pipeline import extract_paper_from_bytes
from bridges_rag.ingest.client import Scraper
from bridges_rag.ingest.models import Paper

logger = logging.getLogger(__name__)


def download_pdf(paper: Paper, data_dir: Path, scraper: Scraper) -> Paper:
    year_dir = data_dir / str(paper.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    dest = year_dir / f"{paper.paper_id}.pdf"

    if not dest.exists():
        response = scraper.get(paper.pdf_url)
        dest.write_bytes(response.content)

    return paper.model_copy(
        update={
            "pdf_path": str(dest.relative_to(data_dir)),
            "pdf_sha256": _sha256_file(dest),
            "pdf_persisted": True,
        }
    )


def harvest_pdf(
    paper: Paper,
    data_dir: Path,
    scraper: Scraper,
    *,
    previous: Paper | None = None,
) -> Paper:
    """Extract markdown straight from downloaded PDF bytes, never writing the PDF to disk.

    Skips the network fetch entirely if the paper's markdown was already extracted, carrying
    forward `previous`'s `pdf_sha256` (from an earlier manifest) so re-runs stay reproducible.
    """
    markdown_dir = data_dir / str(paper.year) / "markdown"
    md_path = markdown_dir / f"{paper.paper_id}.md"

    if md_path.exists():
        sha256 = previous.pdf_sha256 if previous is not None else None
        return paper.model_copy(
            update={"pdf_path": None, "pdf_sha256": sha256, "pdf_persisted": False}
        )

    data = scraper.get(paper.pdf_url).content
    paper = paper.model_copy(
        update={
            "pdf_path": None,
            "pdf_sha256": hashlib.sha256(data).hexdigest(),
            "pdf_persisted": False,
        }
    )

    extracted = extract_paper_from_bytes(paper, data, markdown_dir)
    if not extracted.ok:
        logger.warning("failed to extract markdown for %s: %s", paper.paper_id, extracted.error)
    return paper


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
