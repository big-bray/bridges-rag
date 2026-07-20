from pathlib import Path

from bridges_rag.ingest.manifest import read_manifest, write_manifest
from bridges_rag.ingest.models import Paper


def _paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        year=2025,
        title="A Paper Title",
        authors=["Ada Lovelace", "Alan Turing"],
        category="Regular Papers",
        first_page=1,
        last_page=8,
        abstract="An abstract.",
        detail_url=f"https://archive.bridgesmathart.org/2025/{paper_id}.html",
        pdf_url=f"https://archive.bridgesmathart.org/2025/{paper_id}.pdf",
        isbn="978-1-938664-51-9",
        issn="1099-6702",
        pdf_sha256="deadbeef",
        pdf_path=f"2025/{paper_id}.pdf",
    )


def test_manifest_round_trip(tmp_path: Path):
    papers = [_paper("bridges2025-1"), _paper("bridges2025-9")]
    manifest_path = tmp_path / "2025" / "manifest.jsonl"

    write_manifest(papers, manifest_path)
    loaded = read_manifest(manifest_path)

    assert loaded == papers


def test_manifest_is_one_json_object_per_line(tmp_path: Path):
    papers = [_paper("bridges2025-1"), _paper("bridges2025-9")]
    manifest_path = tmp_path / "manifest.jsonl"

    write_manifest(papers, manifest_path)

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_page_count_included_in_manifest(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest([_paper("bridges2025-1")], manifest_path)

    loaded = read_manifest(manifest_path)
    assert loaded[0].page_count == 8
