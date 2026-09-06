"""Zotero as a document source: resolve items, find their PDFs, save notes.

Unlike the Ollama original, this reads the PDF file itself rather than
Zotero's server-side fulltext index whenever OCR is in play — the index is
plain text with no page structure, so there is no way to tell from it which
pages lack a text layer. The index stays available as a fast path for runs
that have OCR switched off entirely.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from pyzotero import zotero

from lmstudio import log

import config

ITEM_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")

# Every Zotero request costs roughly the same couple of seconds whatever it
# returns, so the page size — not the amount of data — decides how long a
# library-wide listing takes. pyzotero's default of 25 turns 2800 items into
# a 70-second wait; 500 does it in under 20.
PAGE_SIZE = 500
SUMMARY_MARKER = "AI Summary:"


class ProcessingError(Exception):
    """A per-item failure that should not abort a whole collection run."""


class ZoteroUnreachable(Exception):
    """Zotero is not answering — almost always because it is not running."""


def unreachable(local: bool) -> ZoteroUnreachable:
    """The message to show when Zotero refuses the connection."""
    if local:
        return ZoteroUnreachable(
            "Cannot reach Zotero's local API at http://localhost:23119.\n"
            "Zotero itself must be running for --zotero-local: open it, and check\n"
            "Settings -> Advanced -> 'Allow other applications on this computer to\n"
            "communicate with Zotero'. Closing its PDF reader tabs is a good idea\n"
            "before a --replace-pdf run, but the application has to stay open."
        )
    return ZoteroUnreachable(
        "Cannot reach the Zotero web API. Check your network connection, or use\n"
        "--zotero-local to read the library from a running Zotero instead."
    )


def probe(local: bool) -> None:
    """Fail fast, and legibly, when Zotero is not there to answer.

    Without this the first API call raises an httpx ConnectError, which
    reaches the user as sixty lines of traceback ending in a localised
    WinError rather than "start Zotero".
    """
    if not local:
        return
    import httpx

    try:
        httpx.get(
            "http://localhost:23119/api/users/0/items",
            params={"limit": 1}, timeout=10,
        )
    except httpx.HTTPError:
        raise unreachable(True) from None


def build_client(local: bool = False) -> zotero.Zotero:
    """A pyzotero client for the web API, or for Zotero's own local API.

    The local API (Zotero 7, Settings -> Advanced -> "Allow other applications
    on this computer to communicate with Zotero") needs no API key and works
    offline, but it is read-only — which is why --zotero-local requires
    --no-note. Zotero must be running for it to answer.
    """
    if local:
        probe(True)
        return zotero.Zotero(
            config.ZOTERO_LIBRARY_ID or "0", config.ZOTERO_LIBRARY_TYPE, local=True
        )

    if not (config.ZOTERO_LIBRARY_ID and config.ZOTERO_API_KEY):
        sys.exit(
            "Missing Zotero credentials. Copy config.example.ini to config.ini and "
            "fill in zotero_library_id and zotero_api_key (get a key at "
            "https://www.zotero.org/settings/keys), or set the ZOTERO_LIBRARY_ID and "
            "ZOTERO_API_KEY environment variables.\n"
            "Alternatively, with Zotero running, use --zotero-local --no-note to read "
            "the local library without any key."
        )
    return zotero.Zotero(
        config.ZOTERO_LIBRARY_ID, config.ZOTERO_LIBRARY_TYPE, config.ZOTERO_API_KEY
    )


# --------------------------------------------------------------------------
# Resolving what to work on
# --------------------------------------------------------------------------

def resolve_item(zot: zotero.Zotero, query: str) -> dict:
    if ITEM_KEY_RE.match(query):
        return zot.item(query)["data"] | {"key": query}

    matches = zot.items(q=query, qmode="titleCreatorYear", itemType="-attachment")
    if not matches:
        sys.exit(f"No Zotero items matched: {query!r}")
    if len(matches) > 1:
        log(f"Multiple matches for {query!r}, pick one and rerun with its key:")
        for m in matches:
            log(f"  {m['key']}  {m['data'].get('title', '(no title)')}")
        sys.exit(1)
    item = matches[0]
    return item["data"] | {"key": item["key"]}


def resolve_collection(zot: zotero.Zotero, query: str) -> str:
    if ITEM_KEY_RE.match(query):
        return query

    collections = zot.everything(zot.collections(limit=PAGE_SIZE))
    matches = [c for c in collections if c["data"]["name"].lower() == query.lower()]
    if not matches:
        matches = [c for c in collections if query.lower() in c["data"]["name"].lower()]
    if not matches:
        sys.exit(f"No collection matched: {query!r}")
    if len(matches) > 1:
        log(f"Multiple collections matched {query!r}, pick one and rerun with its key:")
        for c in matches:
            log(f"  {c['key']}  {c['data']['name']}")
        sys.exit(1)
    return matches[0]["key"]


def _papers_from_items(items: list[dict]) -> list[dict]:
    return [
        {"key": it["key"], "title": it["data"].get("title", "Untitled")}
        for it in items
        if it["data"].get("itemType") not in ("attachment", "note")
    ]


def get_collection_papers(
    zot: zotero.Zotero, collection_key: str, recursive: bool = False
) -> list[dict]:
    """Top-level items in a collection, optionally including its subcollections."""
    papers = _papers_from_items(
        zot.everything(zot.collection_items_top(collection_key, limit=PAGE_SIZE))
    )
    if recursive:
        seen = {p["key"] for p in papers}
        for child in zot.collections_sub(collection_key):
            for paper in get_collection_papers(zot, child["key"], recursive=True):
                if paper["key"] not in seen:
                    seen.add(paper["key"])
                    papers.append(paper)
    return papers


def get_all_papers(zot: zotero.Zotero) -> list[dict]:
    """Every top-level paper in the library: all collections plus loose items."""
    return _papers_from_items(zot.everything(zot.top(limit=PAGE_SIZE)))


# --------------------------------------------------------------------------
# Existing summaries
# --------------------------------------------------------------------------

def _summary_has_body(note_html: str) -> bool:
    """True if a summary note contains real text beyond the AI Summary header."""
    without_header = re.sub(r"<h1>.*?</h1>", " ", note_html, flags=re.DOTALL)
    body_text = re.sub(r"<[^>]+>", " ", without_header)
    return bool(body_text.strip())


def find_summary_notes(zot: zotero.Zotero, parent_key: str) -> list[dict]:
    """All child AI Summary notes of an item, as full API objects (delete-able)."""
    return [
        child
        for child in zot.children(parent_key)
        if child["data"].get("itemType") == "note"
        and SUMMARY_MARKER in child["data"].get("note", "")
    ]


def has_existing_summary(zot: zotero.Zotero, parent_key: str) -> bool:
    # Blank summaries (header but no body) left behind by earlier runs where the
    # model returned an empty response don't count, so a plain rerun retries
    # them instead of skipping.
    return any(
        _summary_has_body(note["data"].get("note", ""))
        for note in find_summary_notes(zot, parent_key)
    )


def get_summarized_keys(zot: zotero.Zotero) -> set[str]:
    """Keys of all items that already have a non-blank AI Summary note.

    One paginated pass over the library's notes, instead of a children()
    request per paper: scanning the whole library would otherwise cost
    thousands of requests before any summarizing starts.
    """
    keys = set()
    for note in zot.everything(zot.items(itemType="note", limit=PAGE_SIZE)):
        data = note["data"]
        parent = data.get("parentItem")
        note_html = data.get("note", "")
        if parent and SUMMARY_MARKER in note_html and _summary_has_body(note_html):
            keys.add(parent)
    return keys


def delete_summary_notes(zot: zotero.Zotero, notes: list[dict], label: str) -> None:
    for note in notes:
        try:
            zot.delete_item(note)
            log(f"  deleted {label} summary note {note['key']}")
        except Exception as exc:
            log(f"  warning: could not delete {label} summary note {note['key']}: {exc}")


def delete_blank_summary_notes(zot: zotero.Zotero, parent_key: str) -> None:
    """Remove leftover AI Summary notes that have a header but no body."""
    blanks = [
        note
        for note in find_summary_notes(zot, parent_key)
        if not _summary_has_body(note["data"].get("note", ""))
    ]
    delete_summary_notes(zot, blanks, "blank")


# --------------------------------------------------------------------------
# Getting at the PDF
# --------------------------------------------------------------------------

def get_pdf_attachments(zot: zotero.Zotero, page_size: int = PAGE_SIZE) -> dict[str, dict]:
    """Map every item key to its PDF attachment, in one paginated sweep.

    Asking children() per item is the obvious way and the wrong one: every
    request to Zotero costs about the same fixed couple of seconds whatever it
    returns, so a library-wide run spends hours doing nothing but waiting.
    Fetching all attachments a few hundred at a time turns thousands of
    requests into a handful.

    Items with several PDFs keep the first, matching find_pdf_attachment().
    """
    index: dict[str, dict] = {}
    for item in zot.everything(zot.items(itemType="attachment", limit=page_size)):
        data = item["data"]
        parent = data.get("parentItem")
        if parent and data.get("contentType") == "application/pdf":
            index.setdefault(parent, data | {"key": item["key"]})
    return index


def find_pdf_attachment(zot: zotero.Zotero, parent_key: str) -> dict:
    for child in zot.children(parent_key):
        data = child["data"]
        if (
            data.get("itemType") == "attachment"
            and data.get("contentType") == "application/pdf"
        ):
            return data | {"key": child["key"]}
    raise ProcessingError(f"No PDF attachment found under item {parent_key}")


def local_pdf_path(attachment: dict) -> Path | None:
    """Where this attachment's PDF lives on disk, if it is there at all.

    Stored files sit in <storage>/<attachment key>/<filename>. Linked files
    keep an explicit path, which Zotero writes either absolute or relative to
    the data directory as "attachments:...".
    """
    link_mode = attachment.get("linkMode", "")
    if link_mode == "linked_file":
        raw = attachment.get("path", "")
        if not raw:
            return None
        if raw.startswith("attachments:"):
            base = Path(config.ZOTERO_STORAGE_DIR).parent
            candidate = base / raw[len("attachments:"):]
        else:
            candidate = Path(raw)
        return candidate if candidate.exists() else None

    filename = attachment.get("filename", "")
    if not filename:
        return None
    candidate = Path(config.ZOTERO_STORAGE_DIR) / attachment["key"] / filename
    return candidate if candidate.exists() else None


def server_fulltext(zot: zotero.Zotero, attachment_key: str) -> str:
    """Zotero's own indexed plain text for an attachment, or "" if it has none."""
    try:
        return (zot.fulltext_item(attachment_key).get("content") or "").strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Saving the summary back
# --------------------------------------------------------------------------

def save_note(zot: zotero.Zotero, parent_key: str, title: str, summary_html: str) -> None:
    # Built by hand rather than via zot.item_template("note"): pyzotero caches
    # that template and, once it is over an hour old, revalidates it with a
    # request that (due to a pyzotero bug) omits the required itemType param,
    # causing a 400.
    note = {
        "itemType": "note",
        "note": f"<h1>{SUMMARY_MARKER} {html.escape(title)}</h1>{summary_html}",
        "tags": [],
        "collections": [],
        "relations": {},
        "parentItem": parent_key,
    }
    result = zot.create_items([note])
    if result.get("failed"):
        log(f"  warning: failed to create Zotero note: {result['failed']}")
    else:
        log("  saved summary as a Zotero note.")
