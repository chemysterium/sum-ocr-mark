"""Write an invisible text layer into a PDF, so scans become searchable.

OCR gives us the words on a scanned page; this puts them back into the file
as invisible text, positioned over the pixels they came from. The page still
looks exactly the same, but Zotero can index it and a reader can search and
select it.

Placement comes from DeepSeek-OCR's <|grounding|> mode, which prefixes each
block with `label[[x1, y1, x2, y2]]` in coordinates normalised to 0-1000
against the page. (Verified against a page with text at a known position:
x=72pt on a 595pt-wide page was reported as 115, matching 72/595*1000 = 121
to within a percent.)

The font matters more than it looks. PyMuPDF derives a PDF's ToUnicode table
by reverse-mapping glyphs, and most fonts map one glyph from several
codepoints — in Times the space glyph comes back as U+00A0 and the hyphen as
U+00AD, which would quietly break phrase search on every space in the
document. pick_font() prefers fonts that survive that round trip.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Fonts worth trying first, chosen for broad Unicode coverage. The list is
# only a starting order: every candidate is round-trip tested below, because
# coverage alone does not predict whether the text comes back out intact.
FONT_CANDIDATES = [
    # Windows. Lucida Sans Unicode and Cascadia round-trip exactly; the
    # everyday UI fonts do not (Segoe and Calibri return U+2010 for a plain
    # hyphen, Times and Arial return U+00AD, and most return U+00A0 for a
    # space) — which would quietly break search on hyphenated terms.
    r"C:\Windows\Fonts\l_10646.ttf",
    r"C:\Windows\Fonts\CascadiaCode.ttf",
    r"C:\Windows\Fonts\SansSerifCollection.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

# Searched when nothing in FONT_CANDIDATES round-trips cleanly.
FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
]

# Characters whose glyphs are commonly shared with a look-alike codepoint,
# so a font that returns these unchanged will not corrupt ordinary search.
ROUNDTRIP_PROBE = "AG50W-X12 at 120 C"

# A block whose box is thinner than this (in points) cannot hold readable
# text and is almost always a stray detection.
MIN_BOX_SIZE = 2.0

# A page already holding this much text has a real text layer, so OCR text
# must not be added on top of it. Kept in step with extract.MIN_PAGE_CHARS.
MIN_EXISTING_CHARS = 80

_BLOCK_HEADER = re.compile(
    r"^[ \t]*([a-z_]+)\[\[([\d\s,]+)\]\][ \t]*$", re.MULTILINE
)


def log(message: str) -> None:
    print(message, file=sys.stderr)


@dataclass
class Block:
    """One OCR-ed region: its text, and where it sits on the page.

    bbox is (x0, y0, x1, y1) normalised to 0-1000; None when the OCR prompt
    returned no coordinates, in which case the caller falls back to spreading
    the text over the whole page.
    """

    text: str
    bbox: tuple[float, float, float, float] | None = None


def parse_grounding(raw: str) -> list[Block]:
    """Split DeepSeek-OCR grounding output into positioned blocks.

    Text before the first header, or output with no headers at all (the plain
    "Free OCR." prompt), comes back as a single block with no box.
    """
    headers = list(_BLOCK_HEADER.finditer(raw))
    if not headers:
        text = raw.strip()
        return [Block(text)] if text else []

    blocks: list[Block] = []
    leading = raw[: headers[0].start()].strip()
    if leading:
        blocks.append(Block(leading))

    for i, match in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        text = raw[match.end() : end].strip()
        if not text:
            continue
        numbers = [int(n) for n in match.group(2).replace(",", " ").split()]
        bbox = None
        if len(numbers) >= 4:
            x0, y0, x1, y1 = numbers[:4]
            bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        blocks.append(Block(text, bbox))
    return blocks


# --------------------------------------------------------------------------
# Font selection
# --------------------------------------------------------------------------

_font_cache: dict[str, object] = {}


def _roundtrips(font, probe: str) -> bool:
    """True if text written with this font extracts back unchanged.

    A PDF records which glyph to draw, and a separate ToUnicode table says
    what each glyph means. PyMuPDF builds that table by reverse-mapping
    glyphs to codepoints, and most fonts point several codepoints at one
    glyph — so a plain hyphen can come back as U+2010 and a space as U+00A0.
    Text like that looks right and fails every search for it, so the only
    reliable test is to write it and read it back.
    """
    import pymupdf

    try:
        doc = pymupdf.open()
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        writer.append((40, 80), probe, font=font, fontsize=11)
        writer.write_text(page, render_mode=3)
        result = page.get_text("text").strip()
        doc.close()
        return result == probe
    except Exception:
        return False


def _candidate_paths():
    yield from FONT_CANDIDATES
    for directory in FONT_DIRS:
        base = Path(directory)
        if base.is_dir():
            yield from (str(p) for p in sorted(base.rglob("*.ttf")))


def pick_font(sample: str = ""):
    """Best available (pymupdf Font, path) for an invisible text layer.

    Prefers a font that both covers `sample` and survives a round trip. A
    font that covers everything but mangles spaces is worse than one that
    drops a rare symbol, so coverage alone does not decide.
    """
    import pymupdf

    if "chosen" in _font_cache:
        return _font_cache["chosen"]

    wanted = {ord(c) for c in sample if not c.isspace()}
    probe = ROUNDTRIP_PROBE
    best = None  # (covered, font, path) — full coverage but imperfect round trip
    seen: set[str] = set()

    for path in _candidate_paths():
        if path in seen or not Path(path).exists():
            continue
        seen.add(path)
        try:
            font = pymupdf.Font(fontfile=path)
        except Exception:
            continue

        covered = sum(1 for cp in wanted if font.has_glyph(cp))
        if wanted and covered < len(wanted):
            continue
        if _roundtrips(font, probe):
            _font_cache["chosen"] = (font, path)
            return font, path
        if best is None:
            best = (covered, font, path)

    if best is not None:
        log(
            f"  note: no font both covers this text and survives a round trip; "
            f"using {Path(best[2]).name}. Hyphens or spaces in the text layer may "
            "differ from the original, which can affect exact-phrase search."
        )
        _font_cache["chosen"] = (best[1], best[2])
        return best[1], best[2]

    raise RuntimeError(
        "No usable TrueType font found for the text layer. Install DejaVu Sans, "
        "or drop --text-layer."
    )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

# Markdown the OCR model adds, which should not end up in the text layer.
_MD_NOISE = re.compile(r"^#{1,6}\s+|\*\*|__|^\s*[-*+]\s+|^\s*\|", re.MULTILINE)


def _plain(text: str) -> str:
    """Strip Markdown decoration so the layer holds the words themselves."""
    import render

    text = render.strip_latex(text)
    text = _MD_NOISE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _fit_textbox(page, rect, text: str, fontname: str, font) -> bool:
    """Write `text` invisibly into `rect`, shrinking until it fits.

    Returns False if even the smallest size overflows, so the caller can fall
    back rather than silently dropping the words.
    """
    size = max(4.0, min(rect.height, 20.0))
    while size >= 3.0:
        # render_mode 3 draws nothing; the text is present but invisible.
        if page.insert_textbox(
            rect, text, fontsize=size, fontname=fontname, fontfile=None,
            render_mode=3, align=0,
        ) >= 0:
            return True
        size -= 0.5
    return False


def write_text_layer(
    source: Path,
    target: Path,
    page_blocks: dict[int, list[Block]],
    min_existing_chars: int = MIN_EXISTING_CHARS,
) -> int:
    """Copy `source` to `target` with OCR text added to the given pages.

    page_blocks maps 1-based page numbers to their OCR blocks. Pages absent
    from it keep whatever text layer they already had, and so do pages that
    already hold at least min_existing_chars of text — that threshold is the
    same one that decided which pages needed OCR, so a page is never given a
    second copy of its own words. Returns the number of pages that gained text.
    """
    import pymupdf

    sample = "".join(b.text for blocks in page_blocks.values() for b in blocks)
    font, font_path = pick_font(sample)
    fontname = "SOMF"  # any name; the file is what matters

    doc = pymupdf.open(str(source))
    written = skipped = 0
    try:
        for number, blocks in sorted(page_blocks.items()):
            if not 1 <= number <= doc.page_count:
                continue
            page = doc[number - 1]

            # --ocr force re-reads pages that already had a text layer. Adding
            # OCR text there would leave the page holding the same words twice,
            # which is worse than leaving it alone: search would match a
            # duplicate and the indexer would see doubled content.
            if len(page.get_text("text").strip()) >= min_existing_chars:
                skipped += 1
                continue

            page.insert_font(fontname=fontname, fontfile=font_path)
            width, height = page.rect.width, page.rect.height

            placed = False
            unplaced: list[str] = []
            for block in blocks:
                text = _plain(block.text)
                if not text:
                    continue
                if block.bbox is None:
                    unplaced.append(text)
                    continue
                x0, y0, x1, y1 = block.bbox
                rect = pymupdf.Rect(
                    x0 / 1000 * width, y0 / 1000 * height,
                    x1 / 1000 * width, y1 / 1000 * height,
                )
                if rect.width < MIN_BOX_SIZE or rect.height < MIN_BOX_SIZE:
                    unplaced.append(text)
                    continue
                # A box sized to the printed glyphs is often a hair too small
                # for the same words in a different font, so give it room.
                rect = pymupdf.Rect(
                    rect.x0, rect.y0,
                    min(rect.x1 + 0.08 * width, width),
                    min(rect.y1 + 0.02 * height, height),
                )
                if _fit_textbox(page, rect, text, fontname, font):
                    placed = True
                else:
                    unplaced.append(text)

            # Anything without a usable box still belongs in the file, so it
            # goes into the page margin area where it can be indexed even
            # though its position is only approximate.
            if unplaced:
                margin = pymupdf.Rect(
                    0.04 * width, 0.04 * height, 0.96 * width, 0.96 * height
                )
                if _fit_textbox(page, margin, " ".join(unplaced), fontname, font):
                    placed = True

            if placed:
                written += 1

        if skipped:
            log(
                f"  {skipped} page(s) already had a text layer and were left "
                "untouched"
            )
    finally:
        target.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(target), garbage=3, deflate=True)
        doc.close()
    return written
