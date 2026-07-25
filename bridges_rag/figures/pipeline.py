"""Orchestrates figure extraction for a downloaded Bridges proceedings year.

Requires PDFs persisted on disk (`ingest --persist-pdf`) — unlike text extraction,
recovering bounding boxes and pixel data needs the actual PDF, not just its markdown.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from bridges_rag.figures.models import Figure
from bridges_rag.figures.pdf import extract_figures
from bridges_rag.figures.store import read_figures, write_figures
from bridges_rag.ingest.manifest import read_manifest
from bridges_rag.ingest.models import Paper

logger = logging.getLogger(__name__)


def extract_paper_figures(paper: Paper, data_dir: Path, figures_dir: Path) -> list[Figure]:
    if paper.pdf_path is None:
        logger.warning(
            "skipping %s: no persisted PDF (re-run ingest with --persist-pdf)", paper.paper_id
        )
        return []

    pdf_path = data_dir / paper.pdf_path
    try:
        extracted = extract_figures(pdf_path, first_page=paper.first_page)
    except Exception:
        logger.exception("failed to extract figures for %s", paper.paper_id)
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for index, fig in enumerate(extracted):
        figure_id = f"{paper.paper_id}-fig{index:03d}"
        image_path = figures_dir / f"{figure_id}.{fig.image_ext}"
        image_path.write_bytes(fig.image_bytes)
        figures.append(
            Figure(
                figure_id=figure_id,
                paper_id=paper.paper_id,
                pdf_page=fig.pdf_page,
                proceedings_page=fig.proceedings_page,
                bbox=fig.bbox,
                caption=fig.caption,
                image_path=str(image_path.relative_to(data_dir)),
                image_ext=fig.image_ext,
                width=fig.width,
                height=fig.height,
                checksum=hashlib.sha256(fig.image_bytes).hexdigest(),
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
            )
        )
    return figures


def extract_figures_year(year: int, data_dir: Path, *, limit: int | None = None) -> list[Figure]:
    manifest_path = data_dir / str(year) / "manifest.jsonl"
    papers = read_manifest(manifest_path)
    if limit is not None:
        papers = papers[:limit]

    figures_dir = data_dir / str(year) / "figures"
    figures_path = data_dir / str(year) / "figures.jsonl"

    previous_by_paper: dict[str, list[Figure]] = {}
    if figures_path.exists():
        for figure in read_figures(figures_path):
            previous_by_paper.setdefault(figure.paper_id, []).append(figure)

    total = len(papers)
    all_figures: list[Figure] = []
    for i, paper in enumerate(papers, start=1):
        previous = previous_by_paper.get(paper.paper_id, [])
        if previous and all((data_dir / fig.image_path).exists() for fig in previous):
            figures, status = previous, "cached"
        else:
            start = time.monotonic()
            figures = extract_paper_figures(paper, data_dir, figures_dir)
            status = f"{time.monotonic() - start:.1f}s"
        print(f"[{i}/{total}] {paper.paper_id} ({status}) {len(figures)} figure(s)", flush=True)
        all_figures.extend(figures)

    write_figures(all_figures, figures_path)
    return all_figures
