import hashlib
from pathlib import Path

import pymupdf

from bridges_rag.figures.pipeline import extract_figures_year, extract_paper_figures
from bridges_rag.figures.store import read_figures
from bridges_rag.ingest.manifest import write_manifest
from bridges_rag.ingest.models import Paper


def _pixmap(color: tuple[int, int, int], size: int) -> pymupdf.Pixmap:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, size, size))
    pix.set_rect(pix.irect, color)
    return pix


def _make_pdf(path: Path, *, num_images: int = 1) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    for i in range(num_images):
        top = 50 + i * 120
        page.insert_image(pymupdf.Rect(50, top, 150, top + 50), pixmap=_pixmap((255, 0, 0), 50))
        page.insert_text((50, top + 60), f"Figure {i + 1}: A red square.")
    doc.save(path)
    doc.close()


def _paper(paper_id: str, *, pdf_path: str | None, first_page: int = 1) -> Paper:
    return Paper(
        paper_id=paper_id,
        year=2025,
        title="A Paper Title",
        authors=["Ada Lovelace"],
        category="Regular Papers",
        first_page=first_page,
        last_page=first_page,
        pdf_url=f"https://archive.bridgesmathart.org/2025/{paper_id}.pdf",
        pdf_path=pdf_path,
    )


def test_extract_paper_figures_writes_image_and_metadata(tmp_path: Path):
    data_dir = tmp_path
    pdf_path = data_dir / "2025" / "bridges2025-1.pdf"
    pdf_path.parent.mkdir(parents=True)
    _make_pdf(pdf_path)

    paper = _paper("bridges2025-1", pdf_path="2025/bridges2025-1.pdf", first_page=29)
    figures_dir = data_dir / "2025" / "figures"

    figures = extract_paper_figures(paper, data_dir, figures_dir)

    assert len(figures) == 1
    figure = figures[0]
    assert figure.figure_id == "bridges2025-1-fig000"
    assert figure.paper_id == "bridges2025-1"
    assert figure.pdf_page == 1
    assert figure.proceedings_page == 29
    assert figure.caption == "Figure 1: A red square."
    assert figure.title == paper.title
    assert figure.authors == paper.authors
    assert figure.year == paper.year

    image_path = data_dir / figure.image_path
    assert image_path.exists()
    assert figure.checksum == hashlib.sha256(image_path.read_bytes()).hexdigest()


def test_extract_paper_figures_skips_paper_without_persisted_pdf(tmp_path: Path):
    paper = _paper("bridges2025-2", pdf_path=None)

    figures = extract_paper_figures(paper, tmp_path, tmp_path / "figures")

    assert figures == []


def test_extract_paper_figures_returns_empty_list_on_corrupt_pdf(tmp_path: Path):
    data_dir = tmp_path
    pdf_path = data_dir / "2025" / "bridges2025-3.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"not actually a pdf")

    paper = _paper("bridges2025-3", pdf_path="2025/bridges2025-3.pdf")

    figures = extract_paper_figures(paper, data_dir, data_dir / "2025" / "figures")

    assert figures == []


def test_extract_figures_year_writes_manifest_for_all_papers(tmp_path: Path):
    data_dir = tmp_path
    for paper_id, num_images in [("bridges2025-1", 2), ("bridges2025-2", 1)]:
        pdf_path = data_dir / "2025" / f"{paper_id}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        _make_pdf(pdf_path, num_images=num_images)

    papers = [
        _paper("bridges2025-1", pdf_path="2025/bridges2025-1.pdf"),
        _paper("bridges2025-2", pdf_path="2025/bridges2025-2.pdf"),
    ]
    write_manifest(papers, data_dir / "2025" / "manifest.jsonl")

    figures = extract_figures_year(2025, data_dir)

    assert len(figures) == 3
    manifest_path = data_dir / "2025" / "figures.jsonl"
    assert manifest_path.exists()
    assert len(read_figures(manifest_path)) == 3


def test_extract_figures_year_reuses_cached_figures_on_rerun(tmp_path: Path):
    data_dir = tmp_path
    pdf_path = data_dir / "2025" / "bridges2025-1.pdf"
    pdf_path.parent.mkdir(parents=True)
    _make_pdf(pdf_path)
    write_manifest(
        [_paper("bridges2025-1", pdf_path="2025/bridges2025-1.pdf")],
        data_dir / "2025" / "manifest.jsonl",
    )

    first_run = extract_figures_year(2025, data_dir)
    pdf_path.unlink()  # if a rerun weren't cached, this would error

    second_run = extract_figures_year(2025, data_dir)

    assert [f.figure_id for f in first_run] == [f.figure_id for f in second_run]
    assert [f.checksum for f in first_run] == [f.checksum for f in second_run]
