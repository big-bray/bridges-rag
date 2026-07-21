from pathlib import Path

from bridges_rag.ingest.client import Scraper
from bridges_rag.ingest.download import harvest_pdf
from bridges_rag.ingest.models import Paper


def _paper(paper_id: str, *, year: int = 2025) -> Paper:
    return Paper(
        paper_id=paper_id,
        year=year,
        title="A Paper Title",
        authors=["Ada Lovelace"],
        category="Regular Papers",
        pdf_url=f"https://archive.bridgesmathart.org/{year}/{paper_id}.pdf",
    )


def test_harvest_pdf_skips_network_fetch_when_markdown_already_exists(tmp_path: Path):
    markdown_dir = tmp_path / "2025" / "markdown"
    markdown_dir.mkdir(parents=True)
    (markdown_dir / "bridges2025-1.md").write_text("already extracted", encoding="utf-8")

    paper = _paper("bridges2025-1")
    previous = paper.model_copy(update={"pdf_sha256": "deadbeef", "pdf_persisted": False})

    # A Scraper whose client is never used to make a request would raise on `.get`, so
    # reaching an assertion here at all proves the network was never touched.
    result = harvest_pdf(paper, tmp_path, Scraper(), previous=previous)

    assert result.pdf_path is None
    assert result.pdf_persisted is False
    assert result.pdf_sha256 == "deadbeef"


def test_harvest_pdf_skip_without_previous_leaves_sha256_none(tmp_path: Path):
    markdown_dir = tmp_path / "2025" / "markdown"
    markdown_dir.mkdir(parents=True)
    (markdown_dir / "bridges2025-1.md").write_text("already extracted", encoding="utf-8")

    result = harvest_pdf(_paper("bridges2025-1"), tmp_path, Scraper(), previous=None)

    assert result.pdf_path is None
    assert result.pdf_sha256 is None
