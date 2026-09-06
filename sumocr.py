"""Summarize or export documents with local models served by LM Studio.

Grown out of doc-ollama-summarizer and zotero-ollama-summarizer: talks to
LM Studio instead of Ollama, and adds OCR for pages that have no text layer.

Pick exactly one source:
    --file PATH                a single document
    --folder PATH              every supported document in a folder
    --zotero-item KEY|TITLE    one Zotero item
    --zotero-collection NAME   every item in a Zotero collection
    --zotero-all               every item in the Zotero library

Pick what to produce with --action:
    summary                    an AI summary (the default)
    markdown                   the document's text as Markdown, no LLM
    both                       both files
    none                       neither, for --text-layer-only runs

And how to handle scanned pages with --ocr:
    auto                       OCR only pages with no text layer (the default)
    force                      OCR every page, ignoring any text layer
    never                      text layer only

--text-layer additionally writes the OCR-ed words back into the PDF as
invisible text, so a scan becomes searchable; --replace-pdf does that to
the original file rather than to a copy.

Examples:
    python sumocr.py --file report.pdf
    python sumocr.py --file scan.pdf --ocr force --action markdown
    python sumocr.py --folder ./papers --style paper --action both
    python sumocr.py --zotero-collection "Thesis Reading" --max-minutes 60
    python sumocr.py --zotero-all --dry-run
    python sumocr.py --zotero-all --action none --text-layer --replace-pdf

Progress goes to stderr, so "sumocr.py --file in.pdf > out.md" captures only
the summary.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import config
import extract
import render
import summarize
from extract import DocumentError, Extraction
from lmstudio import LMStudio, LMStudioError, OCR_PROMPTS, log

SUFFIX_BY_FORMAT = {"md": ".md", "txt": ".txt", "html": ".html"}


class SkipDocument(Exception):
    """This document needs nothing doing — not an error.

    A run that only adds text layers has plenty of legitimate no-ops: a PDF
    that is already searchable, an item whose PDF was never attached, an
    attachment that is not synced to this machine. None of those is a
    failure, and reporting them as failures would bury the real ones in a
    sweep over a whole library.
    """


# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------

def safe_stem(name: str, limit: int = 90) -> str:
    """A filename stem that survives Windows, from an arbitrary title."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned[:limit].rstrip() or "untitled")


def zotero_stem(title: str, key: str) -> str:
    """Filename stem for a Zotero item: a trimmed title plus its item key.

    The key is appended after trimming, never before. Truncating
    "<title> (KEY)" as one string drops the key off the end of any long
    title — losing the one part that makes the name unique, so two papers
    with a similar opening would overwrite each other.
    """
    return f"{safe_stem(title, limit=80)} ({key})"


def markdown_path(stem: str, directory: Path, source: Path | None) -> Path:
    """Where the extracted-text export goes.

    Never overwrite the input: exporting the text of notes.md would otherwise
    replace notes.md with itself.
    """
    candidate = directory / f"{stem}.md"
    if source and candidate.resolve() == source.resolve():
        return directory / f"{stem}.extracted.md"
    return candidate


def summary_path(stem: str, directory: Path, fmt: str) -> Path:
    return directory / f"{stem}.summary{SUFFIX_BY_FORMAT[fmt]}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log(f"  wrote {path}")


# --------------------------------------------------------------------------
# The one operation everything else feeds
# --------------------------------------------------------------------------

class Job:
    """Settings shared by every document in a run."""

    def __init__(self, args: argparse.Namespace, client: LMStudio) -> None:
        self.args = args
        self.client = client
        self.language = summarize.resolve_language(args.language)
        self.ocr_prompt = OCR_PROMPTS[args.ocr_prompt]
        self.chunk_chars = args.chunk_chars
        self.wants_summary = args.action in ("summary", "both")
        self.wants_markdown = args.action in ("markdown", "both")
        # A run whose only product is the text layer never needs the document's
        # text, which lets it skip whole files cheaply — see skip_if_done().
        self.text_layer_only = (
            args.text_layer and not self.wants_summary and not self.wants_markdown
        )
        self.tesseract_layer = (
            args.text_layer and args.text_layer_engine == "tesseract"
        )
        # With Tesseract writing the layer, nothing needs DeepSeek's block
        # coordinates, so the plain prompt — which reads better — is used
        # for the Markdown and summaries.
        self.needs_lmstudio = self.wants_summary or self.wants_markdown
        # Set by a batch run to every item's PDF attachment, fetched in one
        # sweep — see pdf_attachment().
        self.attachments: dict[str, dict] | None = None

    def pdf_attachment(self, zot, key: str) -> dict:
        """An item's PDF attachment, from the prefetched index when there is one.

        Every request to Zotero costs a fixed couple of seconds whatever it
        returns, so a batch resolves all attachments up front rather than
        asking per item.
        """
        import zotero_source

        if self.attachments is None:
            return zotero_source.find_pdf_attachment(zot, key)
        attachment = self.attachments.get(key)
        if attachment is None:
            raise zotero_source.ProcessingError(
                f"No PDF attachment found under item {key}"
            )
        return attachment

    def skip_if_done(self, path: Path) -> None:
        """Raise SkipDocument when a text-layer run has nothing to add.

        Checked before any extraction: reading the page text is cheap, while
        the Markdown pass over a 100-page PDF is not, and in a library sweep
        almost every file lands here.
        """
        if not self.text_layer_only or self.args.ocr == "force":
            return
        if path.suffix.lower() != ".pdf":
            raise SkipDocument("not a PDF, so it has no text layer to add")
        total, empty = extract.pdf_page_report(path, self.args.min_page_chars)
        if not empty:
            raise SkipDocument(f"all {total} page(s) already have a text layer")

    def extract(self, path: Path) -> Extraction:
        return extract.extract(
            path,
            client=self.client,
            ocr_mode=self.args.ocr,
            ocr_model=self.args.ocr_model,
            ocr_prompt=self.ocr_prompt,
            dpi=self.args.ocr_dpi,
            min_page_chars=self.args.min_page_chars,
            want_boxes=self.args.text_layer and not self.tesseract_layer,
            need_text=not self.text_layer_only,
        )

    def write_text_layer(
        self, path: Path, extraction: Extraction | None, directory: Path | None
    ) -> None:
        """Make `path` searchable, with whichever engine was chosen.

        Tesseract is asked for the whole PDF and places every word itself.
        DeepSeek can only supply block positions, so that path needs the
        extraction its OCR pass produced.
        """
        import textlayer

        if not self.args.text_layer or path.suffix.lower() != ".pdf":
            return

        target = path if self.args.replace_pdf else (
            (directory or path.parent) / f"{path.stem}.ocr.pdf"
        )

        def produce(destination: Path) -> int:
            if self.tesseract_layer:
                return textlayer.add_text_layer_with_tesseract(
                    path, destination, self.args.ocr, self.args.ocr_lang,
                    self.args.min_page_chars,
                )
            if extraction is None or not extraction.ocr_blocks:
                return 0
            return textlayer.write_text_layer(
                path, destination, extraction.ocr_blocks, self.args.min_page_chars
            )

        if not self.tesseract_layer and (extraction is None or not extraction.ocr_blocks):
            if extraction is not None and extraction.used_ocr:
                log("  no positioned OCR output, skipping the text layer")
            else:
                log("  no pages needed OCR, so the PDF already has a text layer")
            return

        if self.args.replace_pdf:
            # Never overwrite the only copy: keep the original next to it, and
            # write through a temporary file so an interrupted save cannot
            # leave a truncated PDF where the attachment used to be.
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
                log(f"  kept the original as {backup.name}")
            staged = path.with_suffix(".ocr-tmp.pdf")
            try:
                pages = produce(staged)
                staged.replace(path)
            finally:
                if staged.exists():
                    staged.unlink()
        else:
            pages = produce(target)

        engine = "tesseract" if self.tesseract_layer else "deepseek"
        log(f"  added a text layer to {pages} page(s) with {engine} -> {target}")

    def summarize(self, extraction: Extraction, style: str) -> str:
        log(
            f"  summarizing with {self.args.model} "
            f"(style: {style}, language: {self.language or 'same as document'})..."
        )
        return summarize.summarize(
            self.client,
            extraction.text,
            style,
            self.args.model,
            language=self.language,
            chunk_chars=self.chunk_chars,
            max_tokens=self.args.max_tokens,
            from_ocr=extraction.used_ocr,
        )


# --------------------------------------------------------------------------
# File and folder sources
# --------------------------------------------------------------------------

def process_file(job: Job, path: Path, directory: Path | None, to_stdout: bool) -> None:
    """Extract one file and write whichever outputs were asked for."""
    job.skip_if_done(path)

    # A Tesseract-only run never needs the model to read the document.
    extraction = None
    if not (job.text_layer_only and job.tesseract_layer):
        log(f"Extracting {path.name}...")
        extraction = job.extract(path)
        log(f"  {extraction.describe()}")
    job.write_text_layer(path, extraction, directory)
    if extraction is None:
        return

    out_dir = directory or path.parent
    stem = job.args.output.stem if job.args.output else path.stem
    style = job.args.style if job.args.style != "auto" else "general"
    results: list[tuple[Path | None, str]] = []

    if job.wants_markdown:
        body = render.to_markdown(extraction.text)
        target = job.args.output if (job.args.output and not job.wants_summary) else None
        results.append((target or markdown_path(stem, out_dir, path), body))

    if job.wants_summary:
        summary = job.summarize(extraction, style)
        body = render.render(summary, job.args.format, path.stem)
        target = job.args.output or (
            None if to_stdout else summary_path(stem, out_dir, job.args.format)
        )
        results.append((target, body))

    for target, body in results:
        if target is None:
            print(body)
        else:
            write(target, body)


def collect_files(folder: Path, recursive: bool) -> list[Path]:
    if not folder.is_dir():
        raise DocumentError(f"Not a folder: {folder}")
    walker = folder.rglob("*") if recursive else folder.glob("*")
    files = [
        p
        for p in sorted(walker)
        if p.is_file()
        and p.suffix.lower() in extract.SUPPORTED_SUFFIXES
        # Skip this tool's own previous output, so a rerun over the same
        # folder does not summarize its own summaries.
        and not p.name.endswith((".summary.md", ".summary.txt", ".summary.html"))
        and not p.name.endswith(".extracted.md")
    ]
    return files


def _output_dir_for(path: Path, folder: Path, out_dir: Path | None) -> Path:
    """Where one folder document's outputs go.

    With --output-dir the source tree is mirrored underneath it, rather than
    flattened into it. Flattening silently loses work: a recursive run over a
    tree with reports/2023/summary.pdf and reports/2024/summary.pdf would
    write both to summary.md, the second overwriting the first, and still
    report two files processed.
    """
    if out_dir is None:
        return path.parent
    try:
        return out_dir / path.parent.relative_to(folder)
    except ValueError:  # not under `folder` — keep it flat rather than fail
        return out_dir


def run_folder(job: Job, folder: Path) -> int:
    files = collect_files(folder, job.args.recursive)
    if not files:
        log(f"No supported documents found in {folder}.")
        return 0
    log(f"Found {len(files)} document(s) in {folder}.")

    out_dir = job.args.output_dir
    processed = skipped = failed = 0
    for i, path in enumerate(files, 1):
        log(f"[{i}/{len(files)}] {path.name}")
        target_dir = _output_dir_for(path, folder, out_dir)

        if job.args.skip_existing:
            done = (
                not job.wants_summary
                or summary_path(path.stem, target_dir, job.args.format).exists()
            ) and (
                not job.wants_markdown
                or markdown_path(path.stem, target_dir, path).exists()
            )
            if done:
                log("  output already exists, skipping (drop --skip-existing to redo)")
                skipped += 1
                continue

        if job.args.dry_run:
            try:
                job.skip_if_done(path)
            except SkipDocument as reason:
                log(f"  would skip: {reason}")
                skipped += 1
                continue
            _report_plan(job, path)
            processed += 1
            continue

        try:
            process_file(job, path, target_dir, to_stdout=False)
            processed += 1
        except SkipDocument as reason:
            log(f"  skipped: {reason}")
            skipped += 1
        except (DocumentError, LMStudioError) as exc:
            log(f"  ERROR: {exc}")
            failed += 1

    verb = "would process" if job.args.dry_run else "processed"
    log(f"Done. {processed} {verb}, {skipped} skipped, {failed} failed.")
    return failed


def _products(job: Job) -> str:
    """What a run would actually write, for --dry-run output."""
    parts = []
    if job.wants_summary:
        parts.append("summary")
    if job.wants_markdown:
        parts.append("markdown")
    if job.args.text_layer:
        parts.append(
            "text layer (in place)" if job.args.replace_pdf else "text layer (copy)"
        )
    return ", ".join(parts) or "nothing"


def _report_plan(job: Job, path: Path) -> None:
    """Say what would happen to a file, without calling any model."""
    if path.suffix.lower() == ".pdf":
        try:
            total, empty = extract.pdf_page_report(path, job.args.min_page_chars)
        except Exception as exc:
            log(f"  would fail to open: {exc}")
            return
        if job.args.ocr == "force":
            plan = f"OCR all {total} page(s)"
        elif job.args.ocr == "auto":
            plan = (
                f"OCR {len(empty)} of {total} page(s) with no text layer"
                if empty
                else f"use the text layer of all {total} page(s)"
            )
        else:
            plan = (
                f"use the text layer; {len(empty)} of {total} page(s) would come "
                "back empty"
                if empty
                else f"use the text layer of all {total} page(s)"
            )
        log(f"  would {plan}, then produce: {_products(job)}")
    else:
        log(f"  would extract text, then produce: {_products(job)}")


# --------------------------------------------------------------------------
# Zotero sources
# --------------------------------------------------------------------------

def zotero_pdf_path(job: Job, zot, key: str) -> Path | None:
    """The item's PDF on this machine, or None if only the server index exists.

    Raises SkipDocument for the cases a text-layer run has no business with:
    an item with no PDF, or one whose file is not synced here.
    """
    import zotero_source

    try:
        attachment = job.pdf_attachment(zot, key)
    except zotero_source.ProcessingError:
        # An item with no PDF — a book record, a web link, a note — is simply
        # not something a text-layer run has any business with.
        if job.text_layer_only:
            raise SkipDocument("no PDF attachment") from None
        raise

    path = zotero_source.local_pdf_path(attachment)
    if path is None:
        # Writing a text layer means writing a file, so an attachment that is
        # not on this machine is a skip, not a fallback.
        if job.text_layer_only:
            raise SkipDocument("PDF is not stored on this machine")
        return None

    job.skip_if_done(path)
    return path


def zotero_server_extraction(job: Job, zot, key: str) -> Extraction:
    """Zotero's own indexed text, for items whose PDF is not on this machine.

    It has no page structure, so OCR and text layers are both impossible on
    it; it is only ever enough to summarize from.
    """
    import zotero_source

    attachment = job.pdf_attachment(zot, key)
    text = zotero_source.server_fulltext(zot, attachment["key"])
    if not text:
        raise zotero_source.ProcessingError(
            f"No local PDF for attachment {attachment['key']} and no indexed "
            f"fulltext on the server. Sync the file, or set zotero_storage_dir "
            f"(currently {config.ZOTERO_STORAGE_DIR})."
        )
    if job.args.ocr != "never":
        log("  warning: PDF not available locally, using Zotero's indexed text "
            "(no OCR possible)")
    return Extraction(text=text)


def process_zotero_item(job: Job, zot, key: str, title: str, replace: bool) -> None:
    import zotero_source

    # Collect the notes to replace up front, but only delete them after the new
    # summary is saved, so a failed run never loses an existing summary.
    old_notes = zotero_source.find_summary_notes(zot, key) if replace else []

    path = zotero_pdf_path(job, zot, key)

    extraction = None
    if path is None:
        extraction = zotero_server_extraction(job, zot, key)
        log(f"  {extraction.describe()}")
    elif not (job.text_layer_only and job.tesseract_layer):
        # A Tesseract-only run never needs the model to read the document.
        log(f"  reading {path.name}")
        extraction = job.extract(path)
        log(f"  {extraction.describe()}")

    if path is not None:
        job.write_text_layer(path, extraction, job.args.output_dir)
    if extraction is None:
        return

    stem = zotero_stem(title, key)
    out_dir = job.args.output_dir or Path.cwd()

    if job.wants_markdown:
        write(markdown_path(stem, out_dir, None), render.to_markdown(extraction.text))

    if job.wants_summary:
        style = job.args.style if job.args.style != "auto" else "paper"
        summary = job.summarize(extraction, style)

        if not job.args.no_note:
            zotero_source.save_note(zot, key, title, render.to_note_html(summary))
            if old_notes:
                zotero_source.delete_summary_notes(zot, old_notes, "old")
            else:
                zotero_source.delete_blank_summary_notes(zot, key)

        if job.args.output or job.args.output_dir or job.args.no_note:
            target = job.args.output or summary_path(stem, out_dir, job.args.format)
            write(target, render.render(summary, job.args.format, title))


def run_zotero_batch(job: Job, zot, papers: list[dict]) -> int:
    import zotero_source

    args = job.args
    # Prefetched in one pass; --force re-summarizes regardless, so skip the scan.
    summarized: set[str] = set()
    if job.wants_summary and not args.no_note and not args.force:
        log("Checking which items already have an AI Summary note...")
        summarized = zotero_source.get_summarized_keys(zot)

    # One sweep for every attachment, instead of a request per item: on a
    # whole library that is the difference between a minute and two hours.
    if len(papers) > 3 and job.attachments is None:
        log("Fetching the attachment list...")
        started = time.monotonic()
        job.attachments = zotero_source.get_pdf_attachments(zot)
        log(
            f"  {len(job.attachments)} PDF attachment(s) in "
            f"{time.monotonic() - started:.0f}s"
        )

    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    processed = skipped = failed = 0
    ran_out_of_time = False

    for i, paper in enumerate(papers, 1):
        # A budget for starting new papers, not a hard timeout: a summary
        # already under way is always allowed to finish and be saved.
        if deadline is not None and time.monotonic() >= deadline:
            log(
                f"Time limit of {args.max_minutes:g} min reached — stopping with "
                f"{len(papers) - i + 1} paper(s) unvisited."
            )
            ran_out_of_time = True
            break

        log(f"[{i}/{len(papers)}] {paper['title']} ({paper['key']})")
        if paper["key"] in summarized:
            log("  already summarized, skipping (use --force to redo)")
            skipped += 1
            continue

        if args.dry_run:
            try:
                attachment = job.pdf_attachment(zot, paper["key"])
            except zotero_source.ProcessingError as exc:
                if job.text_layer_only:
                    log("  would skip: no PDF attachment")
                    skipped += 1
                else:
                    log(f"  would fail: {exc}")
                    failed += 1
                continue

            path = zotero_source.local_pdf_path(attachment)
            if path is None:
                if job.text_layer_only:
                    log("  would skip: PDF is not stored on this machine")
                    skipped += 1
                else:
                    log("  would use Zotero's indexed fulltext (no local PDF)")
                    processed += 1
                continue
            try:
                job.skip_if_done(path)
            except SkipDocument as reason:
                log(f"  would skip: {reason}")
                skipped += 1
                continue
            _report_plan(job, path)
            processed += 1
            continue

        try:
            process_zotero_item(job, zot, paper["key"], paper["title"], args.force)
            processed += 1
        except SkipDocument as reason:
            log(f"  skipped: {reason}")
            skipped += 1
        except Exception as exc:
            log(f"  ERROR: {exc}")
            failed += 1

    verb = "would process" if args.dry_run else "processed"
    log(f"Done. {processed} {verb}, {skipped} skipped, {failed} failed.")
    if ran_out_of_time:
        log("Rerun the same command to continue where this left off.")
    return failed


def run_zotero(job: Job) -> int:
    import zotero_source

    args = job.args
    zot = zotero_source.build_client(local=args.zotero_local)

    if args.zotero_item:
        log(f"Resolving item: {args.zotero_item}")
        item = zotero_source.resolve_item(zot, args.zotero_item)
        key, title = item["key"], item.get("title", "Untitled")
        log(f"Found: {title} ({key})")

        already_done = (
            job.wants_summary
            and not args.no_note
            and not args.force
            and zotero_source.has_existing_summary(zot, key)
        )
        if args.dry_run:
            # Reported inline rather than through run_zotero_batch, whose
            # prefetch would scan every note in the library for one item.
            if already_done:
                log("  already summarized, would skip (use --force to redo)")
                return 0
            try:
                attachment = job.pdf_attachment(zot, key)
                path = zotero_source.local_pdf_path(attachment)
                if path:
                    _report_plan(job, path)
                else:
                    log("  would use Zotero's indexed fulltext (no local PDF)")
            except zotero_source.ProcessingError as exc:
                log(f"  would fail: {exc}")
                return 1
            return 0

        if already_done:
            log("Already summarized (use --force to redo).")
            return 0
        try:
            process_zotero_item(job, zot, key, title, args.force)
        except SkipDocument as reason:
            log(f"  skipped: {reason}")
        except (zotero_source.ProcessingError, DocumentError, LMStudioError) as exc:
            sys.exit(f"Error: {exc}")
        return 0

    if args.zotero_all:
        log("Listing every paper in the library...")
        papers = zotero_source.get_all_papers(zot)
        log(f"Found {len(papers)} papers in the library.")
    else:
        collection_key = zotero_source.resolve_collection(zot, args.zotero_collection)
        papers = zotero_source.get_collection_papers(
            zot, collection_key, recursive=args.recursive
        )
        log(f"Found {len(papers)} papers in collection {collection_key}.")

    return run_zotero_batch(job, zot, papers)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sumocr.py",
        description="Summarize or export documents with local LM Studio models, "
        "OCR-ing scanned pages as needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Progress is written to stderr, so 'sumocr.py --file in.pdf > out.md' "
        "captures only the summary.",
    )

    source = parser.add_argument_group("source (choose exactly one)")
    source.add_argument("--file", type=Path, metavar="PATH", help="A single document")
    source.add_argument(
        "--folder", type=Path, metavar="PATH",
        help="Every supported document in a folder (see --recursive)",
    )
    source.add_argument(
        "--zotero-item", metavar="KEY|TITLE",
        help="One Zotero item, by 8-character key or title search",
    )
    source.add_argument(
        "--zotero-collection", metavar="NAME|KEY",
        help="Every item in a Zotero collection (see --recursive)",
    )
    source.add_argument(
        "--zotero-all", action="store_true",
        help="Every item in the Zotero library, collections and loose items alike",
    )
    source.add_argument(
        "--recursive", "-r", action="store_true",
        help="Descend into subfolders (--folder) or subcollections "
        "(--zotero-collection)",
    )

    what = parser.add_argument_group("what to produce")
    what.add_argument(
        "--action", choices=["summary", "markdown", "both", "none"], default="summary",
        help="summary: an AI summary. markdown: the document text as Markdown, no "
        "LLM. both: both files. none: write no summary or Markdown at all, for "
        "runs whose only purpose is --text-layer. Default: summary",
    )
    what.add_argument(
        "--style", choices=["auto", "general", "paper"], default="auto",
        help="Summary shape. paper gives background/methods/findings/limitations. "
        "Default: auto — paper for Zotero items, general for files and folders",
    )
    what.add_argument(
        "--format", "-f", choices=["md", "txt", "html"], default="md",
        help="Format of the summary output (default: md)",
    )
    what.add_argument(
        "--language", "-l", default="auto", metavar="LANG",
        help="Language of the summary: 'auto' (default) matches the document, or "
        "force one with e.g. en, cs, de, or a language name like Polish",
    )

    ocr = parser.add_argument_group("OCR")
    ocr.add_argument(
        "--ocr", choices=["auto", "force", "never"], default="auto",
        help="auto: OCR only pages whose text layer is missing or too thin. "
        "force: OCR every page. never: text layer only. Default: auto",
    )
    ocr.add_argument(
        "--ocr-dpi", type=int, default=200, metavar="N",
        help="Resolution pages are rendered at before OCR (default: 200; try 300 "
        "for small print)",
    )
    ocr.add_argument(
        "--ocr-prompt", choices=sorted(OCR_PROMPTS), default="free",
        help="free: plain reading-order text (default). markdown: layout-aware, "
        "slower, needs more cleanup",
    )
    ocr.add_argument(
        "--text-layer", action="store_true",
        help="Write the OCR-ed text back into the PDF as an invisible text layer, "
        "so the scan becomes searchable and Zotero can index it. Saves a copy as "
        "<name>.ocr.pdf; forces the grounding OCR prompt, which supplies the "
        "positions",
    )
    ocr.add_argument(
        "--text-layer-engine", choices=["tesseract", "deepseek"], default="tesseract",
        help="Which OCR engine writes the text layer. tesseract (default) places "
        "each word at its true position, so selecting and highlighting work. "
        "deepseek only knows block positions and drifts, but needs no local "
        "install. Summaries and Markdown always come from the LM Studio model",
    )
    ocr.add_argument(
        "--ocr-lang", default="eng", metavar="LANG",
        help="Tesseract language(s) for the text layer, joined with + "
        "(e.g. ces, eng+ces). Default: eng",
    )
    ocr.add_argument(
        "--replace-pdf", action="store_true",
        help="With --text-layer, overwrite the original PDF in place instead of "
        "writing a copy — for Zotero this makes the attachment itself searchable. "
        "The original is kept alongside as <name>.pdf.bak",
    )
    ocr.add_argument(
        "--min-page-chars", type=int, default=extract.MIN_PAGE_CHARS, metavar="N",
        help=f"A page with fewer than N characters counts as having no text layer "
        f"(default: {extract.MIN_PAGE_CHARS})",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output", "-o", type=Path, metavar="PATH",
        help="Write to this exact file (single-document sources only)",
    )
    out.add_argument(
        "--output-dir", type=Path, metavar="DIR",
        help="Write outputs into this folder instead of beside each source",
    )
    out.add_argument(
        "--zotero-local", action="store_true",
        help="Read the library through Zotero's own local API instead of the web "
        "API: no API key, works offline, but read-only, so it requires --no-note "
        "(Zotero must be running, with the local API enabled in Advanced settings)",
    )
    out.add_argument(
        "--no-note", action="store_true",
        help="Zotero sources: do not write the summary back as a Zotero note "
        "(implies writing a file instead)",
    )
    out.add_argument(
        "--skip-existing", action="store_true",
        help="--folder: skip documents whose output files already exist",
    )
    out.add_argument(
        "--force", action="store_true",
        help="Zotero sources: re-summarize items that already have an AI Summary "
        "note, replacing it (deleted only after the new summary is saved)",
    )

    models = parser.add_argument_group("models")
    models.add_argument(
        "--model", default=config.CHAT_MODEL,
        help=f"LM Studio model used for summaries (default: {config.CHAT_MODEL})",
    )
    models.add_argument(
        "--ocr-model", default=config.OCR_MODEL,
        help=f"LM Studio vision model used for OCR (default: {config.OCR_MODEL})",
    )
    models.add_argument(
        "--lmstudio-url", default=config.LMSTUDIO_URL,
        help=f"LM Studio server URL (default: {config.LMSTUDIO_URL})",
    )
    models.add_argument(
        "--chunk-chars", type=int, metavar="N",
        help="Characters per chunk for long documents (default: derived from the "
        "context length the model is loaded with)",
    )
    models.add_argument(
        "--max-tokens", type=int, default=2048, metavar="N",
        help="Maximum tokens per summary response (default: 2048)",
    )
    models.add_argument(
        "--timeout", type=int, default=900, metavar="SEC",
        help="Seconds to wait for one LM Studio response (default: 900)",
    )

    run = parser.add_argument_group("run control")
    run.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done — including which pages would be OCR-ed — "
        "without calling a model or writing anything",
    )
    run.add_argument(
        "--max-minutes", type=float, metavar="N",
        help="Zotero batches: stop starting new papers after N minutes. Rerun to "
        "continue where it left off.",
    )
    return parser


def validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    sources = [
        bool(args.file), bool(args.folder), bool(args.zotero_item),
        bool(args.zotero_collection), args.zotero_all,
    ]
    if sum(sources) != 1:
        parser.error(
            "provide exactly one source: --file, --folder, --zotero-item, "
            "--zotero-collection or --zotero-all"
        )
    if args.output and (args.folder or args.zotero_collection or args.zotero_all):
        parser.error("--output names a single file; use --output-dir for batches")
    if args.output and args.action == "both":
        parser.error("--action both writes two files; use --output-dir instead of --output")
    if args.zotero_local and not args.no_note and args.action in ("summary", "both"):
        parser.error(
            "Zotero's local API is read-only, so --zotero-local cannot save notes: "
            "add --no-note (and --output-dir) to write summaries to files instead"
        )
    if args.replace_pdf and not args.text_layer:
        parser.error("--replace-pdf only makes sense with --text-layer")
    if args.action == "none" and not args.text_layer:
        parser.error(
            "--action none produces nothing on its own; add --text-layer to make "
            "the OCR-ed PDFs searchable"
        )
    if args.text_layer and args.ocr == "never":
        parser.error("--text-layer needs OCR; drop --ocr never")
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be greater than 0")
    if args.ocr_dpi < 72:
        parser.error("--ocr-dpi must be at least 72")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)

    client = LMStudio(args.lmstudio_url, timeout=args.timeout)
    needs_summary = args.action in ("summary", "both")

    try:
        job = Job(args, client)

        if not args.dry_run:
            # Fail on a wrong model name now, not after an hour of OCR — but
            # only for the models this run will actually use. A Tesseract-only
            # text-layer run never touches LM Studio at all.
            if job.wants_summary:
                client.check_model(args.model, "summary")
            if job.needs_lmstudio and args.ocr != "never":
                client.check_model(args.ocr_model, "OCR")
            if job.tesseract_layer:
                import textlayer

                if textlayer.find_tesseract() is None:
                    raise DocumentError(
                        "--text-layer-engine tesseract needs Tesseract installed.\n"
                        "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                        "Debian/Ubuntu: apt install tesseract-ocr\n"
                        "macOS: brew install tesseract\n"
                        "Or pass --text-layer-engine deepseek."
                    )

        if args.file:
            if args.dry_run:
                log(args.file.name)
                _report_plan(job, args.file)
            else:
                try:
                    process_file(
                        job,
                        args.file,
                        args.output_dir,
                        # A lone summary with nowhere to write goes to stdout,
                        # as the original doc summarizer did. A Markdown export
                        # is a file by definition and always lands on disk.
                        to_stdout=not (args.output or args.output_dir),
                    )
                except SkipDocument as reason:
                    log(f"  skipped: {reason}")
            failures = 0
        elif args.folder:
            failures = run_folder(job, args.folder)
        else:
            failures = run_zotero(job)
    except (DocumentError, LMStudioError) as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
