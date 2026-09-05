"""Checks for the pure functions — no LM Studio, no Zotero, no network.

Run with `python test_units.py` (or `pytest test_units.py`).
"""

from pathlib import Path

import extract
import render
import sumocr
import summarize


def test_ocr_cleanup_strips_grounding_scaffolding():
    raw = (
        "sub_title[[115, 96, 252, 122]]\n"
        "## Methods\n\n"
        "text[[114, 151, 642, 191]]\n"
        r"Samples were digested in HF-HNO \( _{3} \)  at 120 degC for 48 h." "\n"
        "<|ref|>table<|/ref|><|det|>[[10, 20, 30, 40]]<|/det|>\n"
        "Yield >99%."
    )
    cleaned = extract.clean_ocr_text(raw)
    assert "[[" not in cleaned
    assert "<|" not in cleaned
    assert "## Methods" in cleaned

    # The subscript span must end up attached to the formula, not floating.
    assert "HF-HNO₃ at 120" in render.to_markdown(cleaned)


def test_page_ranges_compress():
    assert extract._ranges([1, 2, 3, 7]) == "1-3, 7"
    assert extract._ranges([5]) == "5"
    assert extract._ranges([1, 2, 4, 5, 6, 9]) == "1-2, 4-6, 9"
    assert extract._ranges([]) == ""


def test_money_is_not_math():
    assert render.to_markdown("costs $5 and $10 per unit") == "costs $5 and $10 per unit"
    converted = render.to_markdown(r"the $\delta^{7}$Li value and $CO_2$")
    assert "δ⁷Li" in converted
    assert "CO₂" in converted


def test_chunking_is_lossless():
    text = "".join(chr(97 + i % 26) for i in range(50000))
    for size in (16896, 5000, 1000):
        chunks = summarize.chunk_text(text, size)
        assert all(len(c) <= size for c in chunks)
        rejoined = chunks[0] + "".join(c[summarize.CHUNK_OVERLAP:] for c in chunks[1:])
        assert rejoined == text, f"text lost at chunk size {size}"
    assert summarize.chunk_text("short", 16896) == ["short"]


def test_language_resolution():
    assert summarize.resolve_language("auto") is None
    assert summarize.resolve_language("CS") == "Czech"
    assert summarize.resolve_language("Brazilian Portuguese") == "Brazilian Portuguese"
    assert "češtině" in summarize.language_reminder("Czech")


def test_filename_safety():
    assert "/" not in sumocr.safe_stem('Li/Be ratios: a "review"?')
    assert sumocr.safe_stem("  ...  ") == "untitled"
    assert len(sumocr.safe_stem("x" * 200)) <= 90


def test_rendering_formats():
    md = "## Findings\n**Bold** and $H_2O$\n* one\n* two"
    plain = render.to_plain_text(md)
    assert "<sub>" not in plain and "H₂O" in plain and "**" not in plain
    assert "<li>" in render.to_html(md, "title")
    assert "<li>" in render.to_note_html(md)


def test_markdown_export_never_overwrites_its_source():
    directory = Path("/tmp")
    assert sumocr.markdown_path("notes", directory, directory / "notes.md").name == (
        "notes.extracted.md"
    )
    assert sumocr.markdown_path("notes", directory, directory / "notes.pdf").name == (
        "notes.md"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
