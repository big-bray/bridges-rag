"""Recovers math glyphs mangled by a broken PDF ToUnicode CMap.

Some math fonts seen in the Bridges corpus (e.g. `NewTXMI`, the txfonts math
italic font) ship a ToUnicode CMap that truncates a Plane-1 codepoint — such as
U+1D703 MATHEMATICAL ITALIC SMALL THETA — to its low 16 bits instead of a UTF-16
surrogate pair, landing the character at U+D703 in the Hangul Syllables block
instead. The same truncation bug lands symbol-font glyphs (e.g. domino tiles) in
the Private Use Area. Adding 0x10000 back deterministically recovers the intended
character; verified against every garbled codepoint found across the 2025 corpus.
"""

from __future__ import annotations

import unicodedata

_TRUNCATED_ASTRAL_RANGES = (
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0xE000, 0xF8FF),  # Private Use Area
)


def _looks_truncated(codepoint: int) -> bool:
    return any(lo <= codepoint <= hi for lo, hi in _TRUNCATED_ASTRAL_RANGES)


def _recover(char: str) -> str:
    codepoint = ord(char)
    if not _looks_truncated(codepoint):
        return char

    candidate = codepoint + 0x10000
    try:
        unicodedata.name(chr(candidate))
    except ValueError:
        return char  # not a recognized character; leave the original alone
    return chr(candidate)


def fix_garbled_math_glyphs(text: str) -> str:
    """Recover math glyphs truncated into the Hangul/PUA blocks by a broken cmap.

    Characters outside the affected ranges, or whose recovered codepoint isn't a
    real assigned character, pass through unchanged — this never raises and never
    touches ordinary text, so it can run unconditionally over extracted markdown.
    """
    return "".join(_recover(char) for char in text)
