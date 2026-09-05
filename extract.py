"""Turn a document into Markdown, OCR-ing the pages that need it.

The interesting case is a PDF that is only partly scanned: a born-digital
paper with a photographed appendix, or a scan whose cover page was added
later. Rather than choosing one strategy for the whole file, each page is
judged on its own text layer — pages that have one are extracted with
pymupdf4llm (which keeps headings and tables intact), pages that do not are
rendered to an image and sent to the OCR model, and the results are stitched
back together in page order.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from lmstudio import LMStudio, OCR_PROMPT_FREE, log

# A page with less than this much text is treated as having no usable text
# layer. Scanned pages usually yield 0 characters, but a scan can carry a
# stray header or a page number from a stamping tool, so the bar sits above 0.
MIN_PAGE_CHARS = 80

# Below this, extraction of the whole document almost certainly failed rather
# than the document being genuinely tiny.
MIN_USEFUL_CHARS = 200

# DeepSeek-OCR is trained around 1024-1280px inputs; rendering much larger
# costs time in the image encoder without improving the transcription.
MAX_RENDER_PIXELS = 1600

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


class DocumentError(Exception):
    """A problem with the input document that the user needs to act on."""


@dataclass
class Extraction:
    """Extracted text plus what it took to get it."""

    text: str
    total_pages: int = 0
    ocr_pages: list[int] = field(default_factory=list)  # 1-based
    source: Path | None = None

    @property
    def used_ocr(self) -> bool:
        return bool(self.ocr_pages)

    def describe(self) -> str:
        if not self.total_pages:
            return f"{len(self.text)} characters"
        note = f"{self.total_pages} page(s), {len(self.text)} characters"
        if self.ocr_pages:
            note += f", {len(self.ocr_pages)} OCR-ed ({_ranges(self.ocr_pages)})"
        return note


def _ranges(pages: list[int]) -> str:
    """Compress [1,2,3,7] into '1-3, 7' for readable progress output."""
    if not pages:
        return ""
    out: list[str] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        out.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    out.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(out)


# --------------------------------------------------------------------------
# OCR output cleanup
# --------------------------------------------------------------------------

# The <|grounding|> prompt prefixes each block with `label[[x, y, x, y]]`, and
# both prompts occasionally leak the special tokens the model is trained on.
_GROUNDING_LINE = re.compile(r"^\s*[a-z_]+\[\[[\d,\s]+\]\]\s*$", re.MULTILINE)
_GROUNDING_INLINE = re.compile(r"<\|(?:ref|/ref|det|/det|grounding)\|>")
_BBOX_INLINE = re.compile(r"\[\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\]")


def clean_ocr_text(text: str) -> str:
    """Strip the layout scaffolding an OCR model emits around real text."""
    text = _GROUNDING_LINE.sub("", text)
    text = _GROUNDING_INLINE.sub("", text)
    text = _BBOX_INLINE.sub("", text)
    text = text.replace("<|image|>", "").replace("<image>", "")
    # Collapse the blank lines the removals leave behind.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def _render_page_png(page, dpi: int) -> bytes:
    """Render a page, backing the DPI off if the result would be oversized."""
    rect = page.rect
    longest = max(rect.width, rect.height) or 1
    if longest * dpi / 72 > MAX_RENDER_PIXELS:
        dpi = max(72, int(MAX_RENDER_PIXELS * 72 / longest))
    return page.get_pixmap(dpi=dpi).tobytes("png")


def pdf_page_report(path: Path, min_page_chars: int = MIN_PAGE_CHARS) -> tuple[int, list[int]]:
    """(page count, 1-based pages with no usable text layer)."""
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        empty = [
            i + 1
            for i, page in enumerate(doc)
            if len(page.get_text("text").strip()) < min_page_chars
        ]
        return doc.page_count, empty


def extract_pdf(
    path: Path,
    client: LMStudio | None = None,
    ocr_mode: str = "auto",
    ocr_model: str = "",
    ocr_prompt: str = OCR_PROMPT_FREE,
    dpi: int = 200,
    min_page_chars: int = MIN_PAGE_CHARS,
) -> Extraction:
    """PDF text as Markdown, OCR-ing pages according to `ocr_mode`.

    ocr_mode is one of:
      never  — text layer only; a scanned page contributes nothing
      auto   — OCR only the pages whose text layer is missing or too thin
      force  — OCR every page, ignoring the text layer entirely
    """
    import pymupdf
    import pymupdf4llm

    with pymupdf.open(str(path)) as doc:
        total = doc.page_count
        if not total:
            raise DocumentError(f"{path.name} has no pages.")

        if ocr_mode == "force":
            ocr_pages = list(range(1, total + 1))
        elif ocr_mode == "auto":
            ocr_pages = [
                i + 1
                for i, page in enumerate(doc)
                if len(page.get_text("text").strip()) < min_page_chars
            ]
        else:
            ocr_pages = []

        if ocr_pages and client is None:
            raise DocumentError("OCR was requested but no LM Studio client was given.")

        text_pages = [n for n in range(1, total + 1) if n not in set(ocr_pages)]

        # Extract every text-layer page in one pymupdf4llm pass. page_chunks
        # returns one entry per requested page, in the order requested, which
        # is what lets the two sources be interleaved below.
        rendered: dict[int, str] = {}
        if text_pages:
            chunks = pymupdf4llm.to_markdown(
                doc,
                pages=[n - 1 for n in text_pages],
                page_chunks=True,
                show_progress=False,
            )
            for number, chunk in zip(text_pages, chunks):
                rendered[number] = (chunk.get("text") or "").strip()

        if ocr_pages:
            log(
                f"  OCR-ing {len(ocr_pages)} of {total} page(s) with {ocr_model} "
                f"({_ranges(ocr_pages)})..."
            )
            started = time.monotonic()
            for position, number in enumerate(ocr_pages, 1):
                png = _render_page_png(doc[number - 1], dpi)
                try:
                    raw = client.ocr_image(png, ocr_model, ocr_prompt)
                except Exception as exc:  # one bad page must not lose the rest
                    log(f"    page {number}: OCR failed ({exc})")
                    rendered[number] = ""
                    continue
                rendered[number] = clean_ocr_text(raw)
                elapsed = time.monotonic() - started
                log(
                    f"    page {number} ({position}/{len(ocr_pages)}): "
                    f"{len(rendered[number])} chars, {elapsed:.0f}s elapsed"
                )

    body = "\n\n".join(rendered[n] for n in range(1, total + 1) if rendered.get(n))
    extraction = Extraction(
        text=body, total_pages=total, ocr_pages=[n for n in ocr_pages if rendered.get(n)],
        source=path,
    )

    if len(body.strip()) < MIN_USEFUL_CHARS:
        empty_count = total - sum(1 for n in range(1, total + 1) if rendered.get(n))
        hint = (
            "Every page came back empty — the PDF is almost certainly scanned. "
            "Re-run with --ocr force."
            if ocr_mode == "never" and empty_count
            else "The OCR model returned almost nothing; try --ocr-dpi 300."
            if ocr_pages
            else "Try --ocr force to read it as images."
        )
        raise DocumentError(
            f"Extracted only {len(body.strip())} characters from {path.name}. {hint}"
        )
    return extraction


# --------------------------------------------------------------------------
# Other formats
# --------------------------------------------------------------------------

def extract_docx(path: Path) -> Extraction:
    """Word text as Markdown, walking the body so tables stay in place.

    python-docx rather than PyMuPDF: PyMuPDF can open .docx but flattens
    tables into loose lines and drops list markers, both of which matter for
    how well the model reads the document.
    """
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    out: list[str] = []
    previous = ""  # kind of the last block: "list", "table" or "text"

    def add(lines: list[str], kind: str) -> None:
        """Append a block, keeping successive list or table rows together."""
        nonlocal previous
        if out and not (kind == previous and kind in ("list", "table")):
            out.append("")
        out.extend(lines)
        previous = kind

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            paragraph = Paragraph(child, doc)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "").lower()
            if style.startswith("heading"):
                level = style.replace("heading", "").strip()
                depth = int(level) if level.isdigit() else 1
                add(["#" * min(depth, 6) + " " + text], "text")
            elif "list bullet" in style:
                add(["- " + text], "list")
            elif "list number" in style:
                add(["1. " + text], "list")
            else:
                add([text], "text")

        elif tag == "tbl":
            table = Table(child, doc)
            rows: list[str] = []
            for i, row in enumerate(table.rows):
                cells = [c.text.strip().replace("|", r"\|") for c in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    rows.append("| " + " | ".join("---" for _ in cells) + " |")
            add(rows, "table")

    return Extraction(text="\n".join(out), source=path)


def extract_text_file(path: Path) -> Extraction:
    return Extraction(text=path.read_text(encoding="utf-8", errors="replace"), source=path)


def extract_image(
    path: Path,
    client: LMStudio | None,
    ocr_model: str,
    ocr_prompt: str = OCR_PROMPT_FREE,
) -> Extraction:
    """A standalone image is always OCR — there is no text layer to prefer."""
    if client is None:
        raise DocumentError(f"{path.name} is an image and can only be read with OCR.")
    log(f"  OCR-ing image with {ocr_model}...")
    text = clean_ocr_text(client.ocr_image(path.read_bytes(), ocr_model, ocr_prompt))
    return Extraction(text=text, total_pages=1, ocr_pages=[1], source=path)


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"} | IMAGE_SUFFIXES


def extract(
    path: Path,
    client: LMStudio | None = None,
    ocr_mode: str = "auto",
    ocr_model: str = "",
    ocr_prompt: str = OCR_PROMPT_FREE,
    dpi: int = 200,
    min_page_chars: int = MIN_PAGE_CHARS,
) -> Extraction:
    """Extract any supported document, OCR-ing as `ocr_mode` directs."""
    if not path.exists():
        raise DocumentError(f"File not found: {path}")
    if path.is_dir():
        raise DocumentError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".doc":
        raise DocumentError(
            "The old binary .doc format is not supported. Open it in Word or "
            "LibreOffice and save as .docx, then try again."
        )

    if suffix == ".pdf":
        result = extract_pdf(
            path, client, ocr_mode, ocr_model, ocr_prompt, dpi, min_page_chars
        )
    elif suffix in IMAGE_SUFFIXES:
        result = extract_image(path, client, ocr_model, ocr_prompt)
    elif suffix == ".docx":
        result = extract_docx(path)
    elif suffix in (".txt", ".md"):
        result = extract_text_file(path)
    else:
        raise DocumentError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix != ".pdf" and len(result.text.strip()) < MIN_USEFUL_CHARS:
        raise DocumentError(
            f"Extracted only {len(result.text.strip())} characters from {path.name}."
        )
    return result
