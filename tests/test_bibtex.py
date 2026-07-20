from bridges_rag.ingest.bibtex import merge_bibtex, parse_bibtex
from bridges_rag.ingest.models import Paper

# Real bibtex file for https://archive.bridgesmathart.org/2025/bridges2025-29.html
REAL_BIBTEX = """@inproceedings{bridges2025:29,
  author      = {Ba\\c{s}e\\u{g}mez, Fahreddin},
  title       = {Algorithmic Generation of Interlaced Stellations},
  pages       = {29--36},
  booktitle   = {Proceedings of Bridges 2025: Mathematics and the Arts},
  year        = {2025},
  editor      = {Verhoeff, Tom and Swart, David and Gould, S. Louise and Torrence, Eve},
  isbn        = {978-1-938664-51-9},
  issn        = {1099-6702},
  publisher   = {Tessellations Publishing},
  address     = {Phoenix, Arizona},

  url         = {http://archive.bridgesmathart.org/2025/bridges2025-29.html}
}
"""


def test_parse_bibtex_editors_publisher_address():
    bibtex = parse_bibtex(REAL_BIBTEX)

    assert bibtex.editors == [
        "Tom Verhoeff",
        "David Swart",
        "S. Louise Gould",
        "Eve Torrence",
    ]
    assert bibtex.publisher == "Tessellations Publishing"
    assert bibtex.address == "Phoenix, Arizona"


def test_parse_bibtex_unescapes_latex_accents_in_editor_names():
    bibtex = parse_bibtex(
        """@inproceedings{bridges2025:29,
  editor      = {Ba\\c{s}e\\u{g}mez, Fahreddin and M\\"{u}ller, Anna},
  publisher   = {Some P\\'{u}blisher},
  address     = {Z\\"{u}rich, Switzerland},
}
"""
    )

    assert bibtex.editors == ["Fahreddin Başeğmez", "Anna Müller"]
    assert bibtex.publisher == "Some Públisher"
    assert bibtex.address == "Zürich, Switzerland"


def test_parse_bibtex_missing_fields_are_none_or_empty():
    bibtex = parse_bibtex("@inproceedings{bridges2025:1,\n  title = {No editor here},\n}\n")

    assert bibtex.editors == []
    assert bibtex.publisher is None
    assert bibtex.address is None


def test_merge_bibtex_fills_in_paper_fields():
    stub = Paper(
        paper_id="bridges2025-29",
        year=2025,
        title="Algorithmic Generation of Interlaced Stellations",
        authors=["Fahreddin Başeğmez"],
        category="Regular Papers",
        first_page=29,
        last_page=36,
        pdf_url="https://archive.bridgesmathart.org/2025/bridges2025-29.pdf",
        bibtex_url="https://archive.bridgesmathart.org/2025/bridges2025-29-bibtex.txt",
    )

    merged = merge_bibtex(stub, REAL_BIBTEX, parse_bibtex(REAL_BIBTEX))

    assert merged.bibtex == REAL_BIBTEX
    assert merged.editors == ["Tom Verhoeff", "David Swart", "S. Louise Gould", "Eve Torrence"]
    assert merged.publisher == "Tessellations Publishing"
    assert merged.address == "Phoenix, Arizona"
    # fields not touched by bibtex parsing are preserved from the stub
    assert merged.title == "Algorithmic Generation of Interlaced Stellations"
    assert merged.first_page == 29
