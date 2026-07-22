from pathlib import Path

import pymupdf
import pytest

from bridges_rag.extract.pdf import extract_pages, extract_pages_from_stream


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


def test_extract_pages_preserves_pdf_page_numbers(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, ["First page of the paper.", "Second page of the paper."])

    pages = extract_pages(pdf_path)

    assert [p.pdf_page for p in pages] == [1, 2]
    assert "First page" in pages[0].markdown
    assert "Second page" in pages[1].markdown


def test_extract_pages_derives_proceedings_page_from_first_page(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, ["Page one.", "Page two.", "Page three."])

    pages = extract_pages(pdf_path, first_page=101)

    assert [p.proceedings_page for p in pages] == [101, 102, 103]


def test_extract_pages_without_first_page_leaves_proceedings_page_none(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, ["Solo page."])

    pages = extract_pages(pdf_path)

    assert pages[0].proceedings_page is None


def test_extract_pages_does_not_crash_on_unicode_math_symbols(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, ["∫ f(x) dx = α + ∇·F ∑ x_i"])

    pages = extract_pages(pdf_path)

    assert len(pages) == 1


def test_extract_pages_raises_on_corrupt_pdf(tmp_path: Path):
    pdf_path = tmp_path / "not_a_pdf.pdf"
    pdf_path.write_bytes(b"this is not a pdf file")

    with pytest.raises(Exception):  # noqa: B017 - exact PyMuPDF exception type is an implementation detail
        extract_pages(pdf_path)


def test_extract_pages_from_stream_matches_path_based_extraction(tmp_path: Path):
    page_texts = ["First page of the paper.", "Second page of the paper."]
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, page_texts)
    data = _make_pdf_bytes(page_texts)

    from_path = extract_pages(pdf_path, first_page=101)
    from_stream = extract_pages_from_stream(data, first_page=101)

    assert [p.model_dump() for p in from_path] == [p.model_dump() for p in from_stream]


def test_extract_pages_from_stream_without_first_page_leaves_proceedings_page_none():
    data = _make_pdf_bytes(["Solo page."])

    pages = extract_pages_from_stream(data)

    assert pages[0].proceedings_page is None


def test_extract_pages_from_stream_raises_on_corrupt_pdf():
    with pytest.raises(Exception):  # noqa: B017 - exact PyMuPDF exception type is an implementation detail
        extract_pages_from_stream(b"this is not a pdf file")
