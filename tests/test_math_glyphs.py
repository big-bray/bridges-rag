from bridges_rag.extract.math_glyphs import fix_garbled_math_glyphs


def test_recovers_math_italic_letters_from_hangul_block():
    # NewTXMI's broken ToUnicode cmap truncates U+1D434 (MATHEMATICAL ITALIC CAPITAL A)
    # to U+D434, which lands in the Hangul Syllables block.
    garbled = "퐴"  # looks like a Hangul syllable, but is really italic A
    assert fix_garbled_math_glyphs(garbled) == "\U0001D434"


def test_recovers_math_italic_greek_letter():
    garbled = "휃"  # truncated U+1D703 MATHEMATICAL ITALIC SMALL THETA
    assert fix_garbled_math_glyphs(garbled) == "\U0001D703"


def test_recovers_symbol_glyph_from_private_use_area():
    garbled = ""  # truncated U+1F062 DOMINO TILE VERTICAL BACK
    assert fix_garbled_math_glyphs(garbled) == "\U0001F062"


def test_leaves_ordinary_text_unchanged():
    text = "The area is A = pi * r^2, see Figure 3."
    assert fix_garbled_math_glyphs(text) == text


def test_leaves_unrecoverable_private_use_character_unchanged():
    # +0x10000 lands on an unassigned codepoint; there's nothing sensible to recover.
    garbled = ""
    assert fix_garbled_math_glyphs(garbled) == garbled


def test_recovers_glyphs_embedded_in_surrounding_markdown():
    garbled = "at angles of _휃_ = {0◦, 90◦}, denoted by _푇푖_"
    fixed = fix_garbled_math_glyphs(garbled)
    assert fixed == (
        "at angles of _\U0001D703_ = {0◦, 90◦}, denoted by _\U0001D447\U0001D456_"
    )


def test_does_not_touch_real_hangul_that_has_no_valid_astral_target():
    # A real Hangul syllable whose codepoint + 0x10000 is unassigned must survive
    # untouched -- the recovery only fires when the candidate is a real character.
    real_hangul = "가"  # 가 (GA), +0x10000 is unassigned
    assert fix_garbled_math_glyphs(real_hangul) == real_hangul
