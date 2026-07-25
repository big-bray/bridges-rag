from pathlib import Path

import pymupdf
import pytest

from bridges_rag.figures.pdf import extract_figures, extract_figures_from_stream


def _pixmap(color: tuple[int, int, int], size: int) -> pymupdf.Pixmap:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, size, size))
    pix.set_rect(pix.irect, color)
    return pix


def _make_pdf(
    path: Path,
    *,
    image_size: int = 50,
    image_rect: pymupdf.Rect | None = None,
    caption: str | None = "Figure 1: A red square.",
    caption_pos: tuple[float, float] = (50, 170),
) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_image(
        image_rect or pymupdf.Rect(50, 50, 150, 150), pixmap=_pixmap((255, 0, 0), image_size)
    )
    if caption is not None:
        page.insert_text(caption_pos, caption)
    doc.save(path)
    doc.close()


def _make_pdf_bytes(**kwargs) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_image(
        kwargs.get("image_rect", pymupdf.Rect(50, 50, 150, 150)),
        pixmap=_pixmap((255, 0, 0), kwargs.get("image_size", 50)),
    )
    caption = kwargs.get("caption", "Figure 1: A red square.")
    if caption is not None:
        page.insert_text(kwargs.get("caption_pos", (50, 170)), caption)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def test_extract_figures_finds_image_with_bbox_and_page(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path)

    figures = extract_figures(pdf_path)

    assert len(figures) == 1
    figure = figures[0]
    assert figure.pdf_page == 1
    assert figure.bbox == (50.0, 50.0, 150.0, 150.0)
    assert figure.width == 50
    assert figure.height == 50
    assert figure.image_ext == "png"
    assert len(figure.image_bytes) > 0


def test_extract_figures_derives_proceedings_page_from_first_page(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path)

    figures = extract_figures(pdf_path, first_page=101)

    assert figures[0].proceedings_page == 101


def test_extract_figures_without_first_page_leaves_proceedings_page_none(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path)

    figures = extract_figures(pdf_path)

    assert figures[0].proceedings_page is None


def test_extract_figures_associates_nearby_figure_caption(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, caption="Figure 1: A red square.")

    figures = extract_figures(pdf_path)

    assert figures[0].caption == "Figure 1: A red square."


def test_extract_figures_leaves_caption_none_when_no_caption_text(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, caption=None)

    figures = extract_figures(pdf_path)

    assert figures[0].caption is None


def test_extract_figures_ignores_caption_too_far_from_image(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, caption="Figure 1: Far away.", caption_pos=(50, 390))

    figures = extract_figures(pdf_path)

    assert figures[0].caption is None


def test_extract_figures_ignores_text_that_is_not_a_figure_label(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, caption="Just some regular paragraph text.")

    figures = extract_figures(pdf_path)

    assert figures[0].caption is None


def test_extract_figures_filters_out_tiny_decorative_images(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, image_size=10, image_rect=pymupdf.Rect(50, 50, 60, 60))

    figures = extract_figures(pdf_path)

    assert figures == []


def test_extract_figures_from_stream_matches_path_based_extraction(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path)
    data = _make_pdf_bytes()

    from_path = extract_figures(pdf_path, first_page=101)
    from_stream = extract_figures_from_stream(data, first_page=101)

    assert len(from_path) == len(from_stream) == 1
    assert from_path[0].bbox == from_stream[0].bbox
    assert from_path[0].caption == from_stream[0].caption
    assert from_path[0].image_bytes == from_stream[0].image_bytes


def test_extract_figures_raises_on_corrupt_pdf(tmp_path: Path):
    pdf_path = tmp_path / "not_a_pdf.pdf"
    pdf_path.write_bytes(b"this is not a pdf file")

    with pytest.raises(Exception):  # noqa: B017 - exact PyMuPDF exception type is an implementation detail
        extract_figures(pdf_path)


def test_extract_figures_from_stream_raises_on_corrupt_pdf():
    with pytest.raises(Exception):  # noqa: B017 - exact PyMuPDF exception type is an implementation detail
        extract_figures_from_stream(b"this is not a pdf file")
