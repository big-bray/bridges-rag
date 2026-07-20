"""Parse a Bridges paper's BibTeX citation file."""

from __future__ import annotations

import re
import unicodedata

from bridges_rag.ingest.models import Paper

_FIELD_START_RE = re.compile(r"(\w+)\s*=\s*\{")

_ACCENT_COMBINING = {
    "'": "́",  # acute
    "`": "̀",  # grave
    "^": "̂",  # circumflex
    '"': "̈",  # diaeresis
    "~": "̃",  # tilde
    "c": "̧",  # cedilla
    "u": "̆",  # breve
    "v": "̌",  # caron
    "H": "̋",  # double acute
    "k": "̨",  # ogonek
    "=": "̄",  # macron
    ".": "̇",  # dot above
    "d": "̣",  # dot below
    "r": "̊",  # ring above
}
_ACCENT_RE = re.compile(r"\\([\"'`^~=.]|[a-zA-Z]+)\{([^{}]*)\}")

_LETTER_MACROS = {
    "o": "ø",
    "O": "Ø",
    "l": "ł",
    "L": "Ł",
    "ae": "æ",
    "AE": "Æ",
    "oe": "œ",
    "OE": "Œ",
    "ss": "ß",
    "aa": "å",
    "AA": "Å",
}
_LETTER_MACRO_RE = re.compile(r"\\(ae|AE|oe|OE|ss|aa|AA|[oOlL])(?![a-zA-Z])")


class BibtexMetadata:
    """Fields scraped from a paper's BibTeX citation file."""

    def __init__(
        self,
        *,
        editors: list[str],
        publisher: str | None,
        address: str | None,
    ) -> None:
        self.editors = editors
        self.publisher = publisher
        self.address = address


def parse_bibtex(text: str) -> BibtexMetadata:
    fields = _parse_fields(text)

    editors_raw = fields.get("editor", "")
    editors = [_bibtex_name_to_plain(name) for name in editors_raw.split(" and ") if name.strip()]

    publisher = fields.get("publisher")
    address = fields.get("address")

    return BibtexMetadata(
        editors=editors,
        publisher=_unescape_latex(publisher).strip() if publisher else None,
        address=_unescape_latex(address).strip() if address else None,
    )


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _FIELD_START_RE.finditer(text):
        key = match.group(1)
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        fields[key] = text[start : i - 1]
    return fields


def _bibtex_name_to_plain(name: str) -> str:
    name = _unescape_latex(name).strip()
    if "," not in name:
        return name
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}"


def _unescape_latex(text: str) -> str:
    text = _ACCENT_RE.sub(_replace_accent, text)
    text = _LETTER_MACRO_RE.sub(lambda m: _LETTER_MACROS[m.group(1)], text)
    return text.replace("{", "").replace("}", "")


def _replace_accent(match: re.Match[str]) -> str:
    command, letters = match.group(1), match.group(2)
    combining = _ACCENT_COMBINING.get(command)
    if combining is None or not letters:
        return letters
    return unicodedata.normalize("NFC", letters[0] + combining) + letters[1:]


def merge_bibtex(paper: Paper, bibtex_text: str, bibtex: BibtexMetadata) -> Paper:
    return paper.model_copy(
        update={
            "bibtex": bibtex_text,
            "editors": bibtex.editors,
            "publisher": bibtex.publisher,
            "address": bibtex.address,
        }
    )
