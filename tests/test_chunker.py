from pathlib import Path

import pytest

from bridges_rag.chunk.chunker import chunk_paper, chunk_year, estimate_tokens
from bridges_rag.extract.models import PageMarkdown
from bridges_rag.extract.pipeline import render_markdown
from bridges_rag.ingest.manifest import write_manifest
from bridges_rag.ingest.models import Paper


def _paper(
    paper_id: str = "bridges2025-1",
    *,
    title: str = "A Paper Title",
    authors: list[str] | None = None,
    year: int = 2025,
) -> Paper:
    return Paper(
        paper_id=paper_id,
        year=year,
        title=title,
        authors=authors if authors is not None else ["Ada Lovelace", "Alan Turing"],
        category="Regular Papers",
        pdf_url=f"https://archive.bridgesmathart.org/2025/{paper_id}.pdf",
    )


def test_chunk_paper_splits_on_headings():
    pages = [
        PageMarkdown(
            pdf_page=1,
            proceedings_page=29,
            markdown="# Introduction\n\nSome intro text.\n\n# Method\n\nSome method text.",
        ),
    ]

    chunks = chunk_paper(_paper(), pages)

    assert [c.heading for c in chunks] == ["# Introduction", "# Method"]
    assert chunks[0].text.startswith("# Introduction")
    assert "Some intro text." in chunks[0].text
    assert chunks[1].text.startswith("# Method")
    assert "Some method text." in chunks[1].text


def test_chunk_paper_content_before_first_heading_has_no_heading():
    pages = [
        PageMarkdown(
            pdf_page=1, proceedings_page=1, markdown="Preamble text.\n\n# Body\n\nBody text."
        ),
    ]

    chunks = chunk_paper(_paper(), pages)

    assert chunks[0].heading is None
    assert chunks[0].text == "Preamble text."
    assert chunks[1].heading == "# Body"


def test_chunk_paper_propagates_paper_metadata_and_pages_across_a_section():
    paper = _paper(title="Tiling the Plane", authors=["Ada Lovelace"], year=2025)
    pages = [
        PageMarkdown(pdf_page=1, proceedings_page=29, markdown="# Intro\n\nText on page one."),
        PageMarkdown(pdf_page=2, proceedings_page=30, markdown="Text continuing on page two."),
    ]

    chunks = chunk_paper(paper, pages)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.title == "Tiling the Plane"
    assert chunk.authors == ["Ada Lovelace"]
    assert chunk.year == 2025
    assert chunk.pdf_pages == [1, 2]
    assert chunk.proceedings_pages == [29, 30]
    assert chunk.paper_id == paper.paper_id


def test_chunk_paper_splits_long_section_with_overlap():
    paragraphs = [f"Sentence number {i} in the section." for i in range(20)]
    pages = [PageMarkdown(pdf_page=1, proceedings_page=1, markdown="\n\n".join(paragraphs))]

    chunks = chunk_paper(_paper(), pages, target_tokens=20, overlap_tokens=8)

    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 20 + estimate_tokens(paragraphs[0])
    # the tail of one chunk reappears at the head of the next (the overlap window)
    assert chunks[0].text.splitlines()[-1] in chunks[1].text


def test_chunk_paper_rejects_overlap_greater_than_or_equal_to_target():
    pages = [PageMarkdown(pdf_page=1, proceedings_page=1, markdown="Some text.")]

    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_paper(_paper(), pages, target_tokens=20, overlap_tokens=20)

    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_paper(_paper(), pages, target_tokens=20, overlap_tokens=30)


def test_chunk_paper_hard_splits_a_single_unbreakable_paragraph():
    huge = "x" * 400  # no sentence punctuation, no page break to split on
    pages = [PageMarkdown(pdf_page=1, proceedings_page=1, markdown=huge)]

    chunks = chunk_paper(_paper(), pages, target_tokens=20, overlap_tokens=5)

    # never crashes, and always makes forward progress even with no split points
    assert len(chunks) > 1
    assert huge.startswith(chunks[0].text.replace("\n\n", ""))
    assert huge.endswith(chunks[-1].text.replace("\n\n", ""))


def test_chunk_paper_hard_split_does_not_split_a_sup_tag():
    # No sentence punctuation, so this falls all the way to the char-count hard split.
    # target_tokens=11 -> chunk_chars=44, which lands squarely inside the <sup> tag
    # (spanning chars 40-53) unless the splitter pushes the cut past it.
    body = "x" * 40 + "<sup>-1</sup>" + "x" * 40
    pages = [PageMarkdown(pdf_page=1, proceedings_page=1, markdown=body)]

    chunks = chunk_paper(_paper(), pages, target_tokens=11, overlap_tokens=2)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.count("<sup>") == chunk.text.count("</sup>")
    assert "<sup>-1</sup>" in "".join(c.text for c in chunks)


def test_chunk_paper_hard_split_does_not_split_an_italic_formula_span():
    body = "x" * 40 + "_formula_" + "x" * 40
    pages = [PageMarkdown(pdf_page=1, proceedings_page=1, markdown=body)]

    chunks = chunk_paper(_paper(), pages, target_tokens=11, overlap_tokens=2)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.count("_") % 2 == 0
    assert "_formula_" in "".join(c.text for c in chunks)


def test_chunk_paper_hard_split_makes_progress_past_an_oversized_protected_span():
    # A protected span longer than the char budget must still force forward progress,
    # not get the splitter stuck re-emitting the same offset.
    body = "<sup>" + "y" * 200 + "</sup>"
    pages = [PageMarkdown(pdf_page=1, proceedings_page=1, markdown=body)]

    chunks = chunk_paper(_paper(), pages, target_tokens=5, overlap_tokens=1)

    assert len(chunks) >= 1
    assert body.startswith(chunks[0].text.replace("\n\n", ""))


def test_chunk_paper_empty_pages_returns_no_chunks():
    assert chunk_paper(_paper(), []) == []


def test_chunk_paper_blank_markdown_returns_no_chunks():
    pages = [PageMarkdown(pdf_page=1, proceedings_page=1, markdown="   \n\n  ")]
    assert chunk_paper(_paper(), pages) == []


def test_chunk_ids_are_sequential_and_unique():
    pages = [
        PageMarkdown(
            pdf_page=1,
            proceedings_page=1,
            markdown="# A\n\nText a.\n\n# B\n\nText b.\n\n# C\n\nText c.",
        ),
    ]

    chunks = chunk_paper(_paper("bridges2025-7"), pages)

    assert [c.chunk_id for c in chunks] == [
        "bridges2025-7-0000",
        "bridges2025-7-0001",
        "bridges2025-7-0002",
    ]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_chunk_year_reads_manifest_and_markdown_and_writes_chunks_jsonl(tmp_path: Path):
    data_dir = tmp_path
    year_dir = data_dir / "2025"
    markdown_dir = year_dir / "markdown"
    markdown_dir.mkdir(parents=True)

    paper_with_markdown = _paper("bridges2025-1")
    paper_without_markdown = _paper("bridges2025-2")
    write_manifest([paper_with_markdown, paper_without_markdown], year_dir / "manifest.jsonl")

    from bridges_rag.extract.models import ExtractedPaper

    extracted = ExtractedPaper(
        paper_id="bridges2025-1",
        pages=[PageMarkdown(pdf_page=1, proceedings_page=29, markdown="# Intro\n\nSome text.")],
    )
    (markdown_dir / "bridges2025-1.md").write_text(render_markdown(extracted), encoding="utf-8")

    chunks = chunk_year(2025, data_dir)

    assert len(chunks) == 1
    assert chunks[0].paper_id == "bridges2025-1"
    assert (year_dir / "chunks.jsonl").exists()
