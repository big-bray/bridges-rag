"""PDF downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

from bridges_rag.ingest.client import Scraper
from bridges_rag.ingest.models import Paper


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
            "pdf_sha256": _sha256(dest),
            "pdf_persisted": True,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
