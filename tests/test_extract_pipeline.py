from pathlib import Path

import pymupdf

from bridges_rag.extract.models import ExtractedPaper, PageMarkdown
from bridges_rag.extract.pipeline import (
    extract_paper,
    extract_paper_from_bytes,
    extract_year,
    parse_pages,
    render_markdown,
)
from bridges_rag.ingest.manifest import write_manifest
from bridges_rag.ingest.models import Paper


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


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


def test_extract_paper_writes_markdown_file(tmp_path: Path):
    data_dir = tmp_path
    pdf_path = data_dir / "2025" / "bridges2025-1.pdf"
    pdf_path.parent.mkdir(parents=True)
    _make_pdf(pdf_path, ["Some paper content."])

    paper = _paper("bridges2025-1", pdf_path="2025/bridges2025-1.pdf", first_page=29)
    markdown_dir = data_dir / "2025" / "markdown"

    extracted = extract_paper(paper, data_dir, markdown_dir)

    assert extracted.ok
    assert extracted.error is None
    md_file = markdown_dir / "bridges2025-1.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "Some paper content." in content
    assert "pdf_page=1" in content
    assert "proceedings_page=29" in content


def test_extract_paper_records_error_instead_of_raising_on_corrupt_pdf(tmp_path: Path):
    data_dir = tmp_path
    pdf_path = data_dir / "2025" / "bridges2025-2.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"not actually a pdf")

    paper = _paper("bridges2025-2", pdf_path="2025/bridges2025-2.pdf")
    markdown_dir = data_dir / "2025" / "markdown"

    extracted = extract_paper(paper, data_dir, markdown_dir)

    assert not extracted.ok
    assert extracted.error is not None
    assert extracted.pages == []
    assert not (markdown_dir / "bridges2025-2.md").exists()


def test_extract_paper_handles_missing_pdf_path(tmp_path: Path):
    paper = _paper("bridges2025-3", pdf_path=None)

    extracted = extract_paper(paper, tmp_path, tmp_path / "markdown")

    assert not extracted.ok
    assert extracted.pages == []


def test_extract_paper_skips_reextraction_when_markdown_already_exists(tmp_path: Path):
    data_dir = tmp_path
    markdown_dir = data_dir / "2025" / "markdown"
    markdown_dir.mkdir(parents=True)
    existing = ExtractedPaper(
        paper_id="bridges2025-4",
        pages=[PageMarkdown(pdf_page=1, proceedings_page=5, markdown="Already extracted.")],
    )
    (markdown_dir / "bridges2025-4.md").write_text(render_markdown(existing), encoding="utf-8")

    # No PDF on disk and no pdf_path in the manifest — if this weren't skipped it would error.
    paper = _paper("bridges2025-4", pdf_path=None)

    extracted = extract_paper(paper, data_dir, markdown_dir)

    assert extracted.ok
    assert extracted.pages == existing.pages


def test_extract_paper_from_bytes_writes_markdown_file_without_persisting_pdf(tmp_path: Path):
    data_dir = tmp_path
    data = _make_pdf_bytes(["Some paper content."])

    paper = _paper("bridges2025-1", pdf_path=None, first_page=29)
    markdown_dir = data_dir / "2025" / "markdown"

    extracted = extract_paper_from_bytes(paper, data, markdown_dir)

    assert extracted.ok
    assert extracted.error is None
    md_file = markdown_dir / "bridges2025-1.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "Some paper content." in content
    assert "pdf_page=1" in content
    assert "proceedings_page=29" in content
    assert not any(data_dir.rglob("*.pdf"))


def test_extract_paper_from_bytes_records_error_instead_of_raising_on_corrupt_pdf(
    tmp_path: Path,
):
    paper = _paper("bridges2025-2", pdf_path=None)
    markdown_dir = tmp_path / "2025" / "markdown"

    extracted = extract_paper_from_bytes(paper, b"not actually a pdf", markdown_dir)

    assert not extracted.ok
    assert extracted.error is not None
    assert extracted.pages == []
    assert not (markdown_dir / "bridges2025-2.md").exists()


def test_render_markdown_joins_pages_with_markers():
    extracted = ExtractedPaper(
        paper_id="bridges2025-1",
        pages=[
            PageMarkdown(pdf_page=1, proceedings_page=29, markdown="Page one text."),
            PageMarkdown(pdf_page=2, proceedings_page=30, markdown="Page two text."),
        ],
    )

    rendered = render_markdown(extracted)

    assert rendered.index("pdf_page=1") < rendered.index("Page one text.")
    assert rendered.index("Page one text.") < rendered.index("pdf_page=2")
    assert rendered.index("pdf_page=2") < rendered.index("Page two text.")


def test_parse_pages_round_trips_render_markdown():
    extracted = ExtractedPaper(
        paper_id="bridges2025-1",
        pages=[
            PageMarkdown(pdf_page=1, proceedings_page=29, markdown="Page one text."),
            PageMarkdown(
                pdf_page=2, proceedings_page=None, markdown="Page two text.\n\nMore text."
            ),
        ],
    )

    pages = parse_pages(render_markdown(extracted))

    assert pages == extracted.pages


def test_extract_year_continues_past_one_bad_paper_and_writes_good_ones(tmp_path: Path):
    data_dir = tmp_path
    good_pdf = data_dir / "2025" / "bridges2025-1.pdf"
    good_pdf.parent.mkdir(parents=True)
    _make_pdf(good_pdf, ["Good paper."])

    bad_pdf = data_dir / "2025" / "bridges2025-2.pdf"
    bad_pdf.write_bytes(b"garbage")

    papers = [
        _paper("bridges2025-1", pdf_path="2025/bridges2025-1.pdf"),
        _paper("bridges2025-2", pdf_path="2025/bridges2025-2.pdf"),
    ]
    write_manifest(papers, data_dir / "2025" / "manifest.jsonl")

    results = extract_year(2025, data_dir)

    assert [r.paper_id for r in results] == ["bridges2025-1", "bridges2025-2"]
    assert results[0].ok
    assert not results[1].ok
    assert (data_dir / "2025" / "markdown" / "bridges2025-1.md").exists()
    assert not (data_dir / "2025" / "markdown" / "bridges2025-2.md").exists()
