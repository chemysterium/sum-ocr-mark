"""Write an invisible text layer into a PDF, so scans become searchable.

OCR gives us the words on a scanned page; this puts them back into the file
as invisible text, positioned over the pixels they came from. The page still
looks exactly the same, but Zotero can index it and a reader can search and
select it.

Two engines can do the placing, and they are not equivalent.

Tesseract (the default, driven through ocrmypdf) reports a box for every
word, which is exactly what a text layer needs. Measured against the ink on
a scanned paper, its words land within 0.33pt of the glyphs at the median
and 4.4pt at worst.

DeepSeek-OCR reports only *block* positions, via its <|grounding|> mode,
which prefixes each block with `label[[x1, y1, x2, y2]]` normalised to
0-1000 against the page. The block text then has to be re-flowed into that
box in a substitute font, so words inside it drift — 3.19pt at the median
and 69.5pt at worst on the same page. It is kept for machines with no
Tesseract installed, and because the same OCR pass also feeds the Markdown.

For the DeepSeek path the font matters more than it looks. PyMuPDF derives a
PDF's ToUnicode table by reverse-mapping glyphs, and most fonts map one
glyph from several codepoints — in Times the space glyph comes back as
U+00A0 and the hyphen as U+00AD, which would quietly break phrase search on
every space in the document. pick_font() prefers fonts that survive that
round trip.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Tesseract ships its own word-level geometry, which is the whole reason to
# prefer it here: measured against the ink on a real scan, its words land
# within 0.3pt at the median and 4.4pt at worst, where placing DeepSeek's
# block text drifts 3.2pt at the median and 69.5pt at worst.
TESSERACT_DIRS = [
    r"C:\Program Files\Tesseract-OCR",
    r"C:\Program Files (x86)\Tesseract-OCR",
    "/usr/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
]

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

# Growth beyond this counts as a page having gained a text layer. Above
# zero because re-saving a PDF can shift a character or two on its own.
MIN_GAINED_CHARS = 5

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


# --------------------------------------------------------------------------
# Tesseract, via ocrmypdf
# --------------------------------------------------------------------------

class TesseractUnavailable(Exception):
    """Tesseract or ocrmypdf is not installed."""


def find_tesseract() -> str | None:
    """Path to the tesseract binary, adding its folder to PATH if needed.

    ocrmypdf shells out to tesseract and only looks on PATH, but on Windows
    the installer does not put it there, so a perfectly good installation
    looks missing. Finding it ourselves avoids telling the user to install
    something they already have.
    """
    found = shutil.which("tesseract")
    if found:
        return found
    for directory in TESSERACT_DIRS:
        candidate = Path(directory) / ("tesseract.exe" if os.name == "nt" else "tesseract")
        if candidate.exists():
            os.environ["PATH"] = f"{os.environ.get('PATH', '')}{os.pathsep}{directory}"
            return str(candidate)
    return None


def tesseract_languages() -> list[str]:
    """Language packs tesseract has installed, or [] if it cannot be asked."""
    binary = find_tesseract()
    if not binary:
        return []
    import subprocess

    try:
        out = subprocess.run(
            [binary, "--list-langs"], capture_output=True, text=True, timeout=30
        )
    except Exception:
        return []
    return [
        line.strip()
        for line in out.stdout.splitlines()[1:]
        if line.strip() and " " not in line.strip()
    ]


def _page_text_lengths(path: Path) -> list[int]:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return [len(page.get_text("text").strip()) for page in doc]


def add_text_layer_with_tesseract(
    source: Path,
    target: Path,
    ocr_mode: str = "auto",
    language: str = "eng",
    rotate: bool = False,
    deskew: bool = False,
) -> int:
    """Make `source` searchable with Tesseract, writing `target`.

    ocrmypdf does the part that is genuinely hard — rasterising each page,
    running Tesseract, and writing the recognised words back at their true
    positions with the right size — so this only maps our OCR modes onto its
    options and reports how many pages gained text.

    Which pages to leave alone is ocrmypdf's own judgement, not this
    tool's MIN_EXISTING_CHARS: skip_text keeps any page that already draws
    text. The two agree in practice — a scan carrying only a library stamp
    is still re-read — and no page can end up with its words twice.
    """
    if find_tesseract() is None:
        raise TesseractUnavailable(
            "Tesseract is not installed, or not where this looked.\n"
            "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Debian/Ubuntu: apt install tesseract-ocr\n"
            "macOS: brew install tesseract\n"
            "Then rerun, or use --text-layer-engine deepseek."
        )
    try:
        import ocrmypdf
    except ImportError:
        raise TesseractUnavailable(
            "The ocrmypdf package is not installed. Run: pip install ocrmypdf\n"
            "(it also needs Ghostscript), or use --text-layer-engine deepseek."
        ) from None

    before = _page_text_lengths(source)

    # --skip-text leaves pages that already carry text exactly as they are,
    # which is the same rule the rest of the tool follows; --force-ocr
    # rasterises and re-reads everything, matching --ocr force.
    options = dict(
        language=language,
        optimize=0,          # keep the run fast; this is not a size exercise
        output_type="pdf",   # plain PDF, not PDF/A: no extra colour profiles
        progress_bar=False,
        rotate_pages=rotate,
        deskew=deskew,
    )
    if ocr_mode == "force":
        options["force_ocr"] = True
    else:
        options["skip_text"] = True

    target.parent.mkdir(parents=True, exist_ok=True)
    # ocrmypdf.ocr() swaps sys.stdout and sys.stderr for StringIO buffers to
    # capture its own logging, and does not put them back. Every later
    # progress line and error message would then be written into a dead
    # buffer — a batch run would go silent after its first OCR-ed file, with
    # no indication anything was wrong. Restore them ourselves.
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        ocrmypdf.ocr(str(source), str(target), **options)
    except Exception as exc:
        raise TesseractUnavailable(
            f"ocrmypdf could not process {source.name}: {exc}"
        ) from None
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr

    # Count pages that actually gained words, rather than pages that crossed
    # some threshold: a sparse page going from nothing to a few dozen
    # characters has still been made searchable, and reporting it as untouched
    # would understate what the run did.
    after = _page_text_lengths(target)
    return sum(
        1
        for old_len, new_len in zip(before, after)
        if new_len > old_len + MIN_GAINED_CHARS
    )
