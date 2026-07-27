"""Heading-aware chunker: splits per-paper markdown into token-budgeted, overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from bridges_rag.chunk.models import Chunk
from bridges_rag.chunk.store import write_chunks
from bridges_rag.extract.models import PageMarkdown
from bridges_rag.extract.pipeline import parse_pages
from bridges_rag.ingest.manifest import read_manifest
from bridges_rag.ingest.models import Paper

DEFAULT_TARGET_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50

_HEADING_RE = re.compile(r"^#{1,6}\s+\S.*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CHARS_PER_TOKEN = 4

# <sup>/<sub> hold exponents and subscripts (e.g. `_f_<sup>-1</sup>`) and single
# underscores wrap italic math variables (e.g. `_\U0001d703_`) -- markdown emitted by
# pymupdf4llm for inline formulas. Splitting inside either garbles the formula, so the
# hard-split fallback below must never cut through one.
_PROTECTED_SPAN_RE = re.compile(r"<su[bp]>.*?</su[bp]>|_[^_\n]+_")


def estimate_tokens(text: str) -> int:
    """Approximate token count without loading a tokenizer (~4 chars/token for English)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class _Paragraph:
    text: str
    pdf_page: int
    proceedings_page: int | None


@dataclass
class _Section:
    heading: str | None
    paragraphs: list[_Paragraph]


def _paragraphs(pages: list[PageMarkdown]) -> list[_Paragraph]:
    paragraphs = []
    for page in pages:
        for block in re.split(r"\n\s*\n", page.markdown):
            block = block.strip()
            if block:
                paragraphs.append(_Paragraph(block, page.pdf_page, page.proceedings_page))
    return paragraphs


def _sections(paragraphs: list[_Paragraph]) -> list[_Section]:
    sections: list[_Section] = []
    current_heading: str | None = None
    current: list[_Paragraph] = []

    def flush() -> None:
        if current:
            sections.append(_Section(current_heading, list(current)))
            current.clear()

    for para in paragraphs:
        if _HEADING_RE.match(para.text):
            flush()
            current_heading = para.text
        else:
            current.append(para)
    flush()
    return sections


def _split_long_paragraph(para: _Paragraph, target_tokens: int) -> list[_Paragraph]:
    """Sub-split a paragraph that alone exceeds the token budget, so packing never emits
    a single chunk unboundedly larger than the target."""
    if estimate_tokens(para.text) <= target_tokens:
        return [para]

    sentences = [s for s in _SENTENCE_RE.split(para.text) if s.strip()]
    if len(sentences) > 1:
        return [_Paragraph(s, para.pdf_page, para.proceedings_page) for s in sentences]

    # No sentence boundaries to split on (e.g. garbled extraction) — hard-split by chars.
    chunk_chars = target_tokens * _CHARS_PER_TOKEN
    cut_points = _safe_cut_points(para.text, chunk_chars)
    return [
        _Paragraph(para.text[start:end], para.pdf_page, para.proceedings_page)
        for start, end in pairwise(cut_points)
    ]


def _safe_cut_points(text: str, chunk_chars: int) -> list[int]:
    """Offsets to hard-split `text` at, pushed forward past any protected span
    (a <sup>/<sub> tag or an underscore-wrapped formula) they would otherwise land inside."""
    protected = [m.span() for m in _PROTECTED_SPAN_RE.finditer(text)]

    points = [0]
    while points[-1] < len(text):
        target = points[-1] + chunk_chars
        for start, end in protected:
            if start < target < end:
                target = end
                break
        points.append(min(target, len(text)))
    return points


def _pack_section(
    section: _Section, target_tokens: int, overlap_tokens: int
) -> list[list[_Paragraph]]:
    """Greedily pack a section's paragraphs into token-budgeted groups, carrying the
    trailing ~overlap_tokens of each group forward into the next."""
    paragraphs = [
        p for para in section.paragraphs for p in _split_long_paragraph(para, target_tokens)
    ]

    groups: list[list[_Paragraph]] = []
    current: list[_Paragraph] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para.text)
        if current and current_tokens + para_tokens > target_tokens:
            groups.append(list(current))

            overlap: list[_Paragraph] = []
            overlap_tokens_used = 0
            for p in reversed(current):
                p_tokens = estimate_tokens(p.text)
                if overlap and overlap_tokens_used + p_tokens > overlap_tokens:
                    break
                overlap.insert(0, p)
                overlap_tokens_used += p_tokens
            current = overlap
            current_tokens = overlap_tokens_used

        current.append(para)
        current_tokens += para_tokens

    if current:
        groups.append(current)
    return groups


def chunk_paper(
    paper: Paper,
    pages: list[PageMarkdown],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    if overlap_tokens >= target_tokens:
        # Otherwise the carried-over overlap alone can already meet or exceed the target,
        # so packing re-exceeds it on every subsequent paragraph, emitting oversized,
        # heavily duplicated groups instead of the intended sliding window.
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than "
            f"target_tokens ({target_tokens})"
        )

    chunks: list[Chunk] = []
    for section in _sections(_paragraphs(pages)):
        for group in _pack_section(section, target_tokens, overlap_tokens):
            body = "\n\n".join(p.text for p in group)
            if not body.strip():
                continue
            text = f"{section.heading}\n\n{body}" if section.heading else body
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}-{len(chunks):04d}",
                    paper_id=paper.paper_id,
                    chunk_index=len(chunks),
                    heading=section.heading,
                    text=text,
                    pdf_pages=sorted({p.pdf_page for p in group}),
                    proceedings_pages=sorted(
                        {p.proceedings_page for p in group if p.proceedings_page is not None}
                    ),
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                )
            )
    return chunks


def chunk_year(
    year: int,
    data_dir: Path,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    manifest_path = data_dir / str(year) / "manifest.jsonl"
    papers = read_manifest(manifest_path)
    markdown_dir = data_dir / str(year) / "markdown"

    all_chunks: list[Chunk] = []
    for paper in papers:
        md_path = markdown_dir / f"{paper.paper_id}.md"
        if not md_path.exists():
            continue
        pages = parse_pages(md_path.read_text(encoding="utf-8"))
        all_chunks.extend(
            chunk_paper(paper, pages, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        )

    write_chunks(all_chunks, data_dir / str(year) / "chunks.jsonl")
    return all_chunks
