"""Checks for the pure functions — no LM Studio, no Zotero, no network.

Run with `python test_units.py` (or `pytest test_units.py`).
"""

from pathlib import Path

import extract
import render
import sumocr
import summarize
import textlayer


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


def test_grounding_blocks_are_parsed_with_boxes():
    raw = (
        "sub_title[[115, 96, 252, 122]]\n"
        "## Methods\n\n"
        "text[[114, 151, 642, 191]]\n"
        "Samples were digested at 120 degC.\n"
    )
    blocks = textlayer.parse_grounding(raw)
    assert len(blocks) == 2
    assert blocks[0].bbox == (115, 96, 252, 122)
    assert blocks[0].text == "## Methods"
    assert blocks[1].bbox == (114, 151, 642, 191)
    assert "digested" in blocks[1].text


def test_grounding_falls_back_when_there_are_no_boxes():
    # The plain "Free OCR." prompt returns no coordinates at all.
    blocks = textlayer.parse_grounding("Just some text.\nOn two lines.")
    assert len(blocks) == 1
    assert blocks[0].bbox is None
    assert textlayer.parse_grounding("   ") == []


def test_grounding_boxes_are_normalised():
    # Reversed corners must come back as a well-formed rectangle.
    blocks = textlayer.parse_grounding("text[[300, 400, 100, 200]]\nhello")
    assert blocks[0].bbox == (100, 200, 300, 400)


def test_text_layer_strips_markdown_decoration():
    assert textlayer._plain("## Heading") == "Heading"
    assert textlayer._plain("**bold** text") == "bold text"
    assert textlayer._plain("- a bullet") == "a bullet"
    assert "₃" in textlayer._plain(r"HF-HNO \( _{3} \)")


def test_output_dir_mirrors_the_source_tree():
    # Same filename in two subfolders must not collide into one output file.
    folder = Path("/docs")
    out = Path("/out")
    a = sumocr._output_dir_for(folder / "2023" / "summary.pdf", folder, out)
    b = sumocr._output_dir_for(folder / "2024" / "summary.pdf", folder, out)
    assert a != b
    assert a == out / "2023" and b == out / "2024"

    # A file directly in the folder lands at the top of the output dir.
    assert sumocr._output_dir_for(folder / "summary.pdf", folder, out) == out

    # With no --output-dir, outputs sit beside each source.
    assert sumocr._output_dir_for(
        folder / "2023" / "summary.pdf", folder, None
    ) == folder / "2023"


def _text_layer_job(**overrides):
    """A Job configured for a text-layer-only run, with no LM Studio behind it."""
    import argparse

    args = argparse.Namespace(
        action="none", text_layer=True, replace_pdf=False, ocr="auto",
        min_page_chars=80, language="auto", ocr_prompt="free", chunk_chars=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return sumocr.Job(args, client=None)


def _pdf(tmp: Path, name: str, texts: list[str]) -> Path:
    """A PDF whose pages carry the given text ("" for a page with no text)."""
    import pymupdf

    doc = pymupdf.open()
    for body in texts:
        page = doc.new_page()
        if body:
            page.insert_text((72, 100), body, fontsize=11)
    path = tmp / name
    doc.save(str(path))
    doc.close()
    return path


def test_text_layer_run_skips_pdfs_that_are_already_searchable():
    import tempfile

    filled = "Lithium isotope fractionation was measured in 24 samples today. " * 3
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        job = _text_layer_job()

        # Every page has text: nothing to add, so the whole file is skipped.
        done = _pdf(tmp, "done.pdf", [filled, filled])
        try:
            job.skip_if_done(done)
            raise AssertionError("a fully searchable PDF should have been skipped")
        except sumocr.SkipDocument as reason:
            assert "already have a text layer" in str(reason)

        # One blank page is enough to make the file worth processing.
        partial = _pdf(tmp, "partial.pdf", [filled, ""])
        job.skip_if_done(partial)  # must not raise

        # --ocr force deliberately re-reads everything, so nothing is skipped.
        _text_layer_job(ocr="force").skip_if_done(done)

        # A run that also wants a summary must not skip on these grounds.
        _text_layer_job(action="summary").skip_if_done(done)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
