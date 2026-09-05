# sum-ocr-mark

Summarize documents, or export their text as Markdown, using local models
served by **LM Studio** — with OCR for the pages that need it.

This grew out of my [doc-ollama-summarizer][doc] and
[zotero-ollama-summarizer][zot]: it merges the two into one tool, swaps
Ollama for LM Studio, and adds per-page OCR.

[doc]: https://github.com/chemysterium/doc-ollama-summarizer
[zot]: https://github.com/chemysterium/zotero-ollama-summarizer

Two models do the work, both served by the same LM Studio instance:

| Role | Default model |
| --- | --- |
| Summaries | `google/gemma-4-26b-a4b-qat` |
| OCR | `deepseek-ocr-2` |

## What is new compared to the earlier two

- **One tool, five sources.** A single file, a folder, one Zotero item, a
  Zotero collection, or the whole Zotero library.
- **Per-page OCR.** A PDF is not classified as "scanned" or "digital" as a
  whole. Each page is judged on its own text layer: pages that have one are
  extracted with pymupdf4llm (headings and tables intact), pages that do not
  are rendered and sent to the OCR model, and the two are stitched back
  together in page order. Partly-scanned PDFs — a digital paper with a
  photographed appendix — come out whole.
- **Markdown export without an LLM.** `--action markdown` runs extraction and
  OCR only, so you can get a clean `.md` of a scanned document without
  summarizing it.
- **Chunk sizes from the live context window.** LM Studio routinely *loads* a
  model with far less context than its weights support (8k on a model capable
  of 262k). The tool reads the loaded context length from LM Studio's API and
  sizes chunks from it, instead of using a constant that silently overflows.
- **Thinking disabled properly.** The earlier two passed Ollama's
  `"think": false` to stop a reasoning model burning its whole output budget
  on hidden tokens and returning nothing. LM Studio's equivalent is
  `reasoning_effort: "none"`, which this sends (with a fallback for older
  builds that reject it).

## Setup

```bash
pip install -r requirements.txt
```

In LM Studio: open the **Developer** tab, click **Start Server**, and load the
summary and OCR models. Then, only if you want the Zotero sources:

```bash
cp config.example.ini config.ini
```

and fill in your Zotero library ID and API key from
<https://www.zotero.org/settings/keys>. Environment variables with the same
uppercased names override the file. Nothing in `config.ini` is needed for
`--file` and `--folder` runs.

## Usage

Pick exactly one source, and optionally what to produce and how to handle
scanned pages:

```bash
# Summarize one document to stdout
python sumocr.py --file report.pdf

# A scanned document: OCR every page, export the text, no summary
python sumocr.py --file scan.pdf --ocr force --action markdown

# A folder of papers: Markdown plus a summary for each, next to the source
python sumocr.py --folder ./papers --recursive --style paper --action both

# One Zotero item, by key or by title search
python sumocr.py --zotero-item ABCD1234
python sumocr.py --zotero-item "lithium isotope fractionation"

# A Zotero collection, including its subcollections, with a time budget
python sumocr.py --zotero-collection "Thesis Reading" -r --max-minutes 60

# The whole library — see what it would do first
python sumocr.py --zotero-all --dry-run
```

### Sources

| Switch | Scope |
| --- | --- |
| `--file PATH` | One document |
| `--folder PATH` | Every supported document in a folder (`-r` to recurse) |
| `--zotero-item KEY\|TITLE` | One Zotero item |
| `--zotero-collection NAME\|KEY` | One collection (`-r` for subcollections) |
| `--zotero-all` | Every top-level item in the library |

Supported files: `.pdf`, `.docx`, `.txt`, `.md`, and images (`.png`, `.jpg`,
`.tif`, `.webp`, `.bmp`), which are always read by OCR.

`--zotero-local` reads the library through Zotero's own local API instead of
the web API — no API key, works offline — but that API is read-only, so it
requires `--no-note` and writes summaries to files:

```bash
python sumocr.py --zotero-collection "Thesis Reading" \
    --zotero-local --no-note --output-dir ./summaries
```

Zotero must be running, with *Settings → Advanced → Allow other applications
on this computer to communicate with Zotero* enabled.

### What to produce

`--action summary` (default) writes an AI summary; `markdown` writes the
extracted text with no LLM involved; `both` writes both files.

`--style paper` shapes the summary as background / methods / findings /
limitations; `general` covers purpose, main points and conclusions. The
default, `auto`, uses `paper` for Zotero items and `general` for files and
folders.

`--format md|txt|html` sets the summary format, and `--language` its language
— `auto` (default) matches the document, or force one with `en`, `cs`, `de`,
or a language name like `Polish`.

### OCR

| `--ocr` | Behaviour |
| --- | --- |
| `auto` (default) | OCR only pages whose text layer is missing or too thin |
| `force` | OCR every page, ignoring any text layer |
| `never` | Text layer only |

A page counts as having no text layer below `--min-page-chars` (default 80),
which is above zero because scanning tools often stamp a page number or header
onto an otherwise blank scan.

`--ocr-dpi` (default 200) sets the render resolution — try 300 for small
print. Pages are capped at 1600 px on the long edge, since DeepSeek-OCR is
trained around 1024–1280 px and larger inputs cost time without reading
better.

`--ocr-prompt free` (default) returns plain reading-order text.
`--ocr-prompt markdown` is layout-aware but emits bounding-box lines and
LaTeX-wrapped subscripts, which are stripped afterwards.

`--dry-run` reports exactly which pages would be OCR-ed, without calling a
model:

```
[3/4] scan.pdf
  would OCR 2 of 3 page(s) with no text layer, then produce: summary
```

### Where output goes

- `--file` with no output switch prints the **summary** to stdout, so
  `sumocr.py --file in.pdf > out.md` works. Progress goes to stderr.
- A **Markdown export** is always a file: `<name>.md` beside the source (or
  `<name>.extracted.md` if that would overwrite the input).
- Summaries written to disk are `<name>.summary.md` / `.txt` / `.html`.
- `--output PATH` names one exact file; `--output-dir DIR` collects a batch.
- Zotero summaries are saved back as a child note named `AI Summary: <title>`.
  Add `--no-note` to get a file instead, or `--output-dir` to get both.

Reruns are cheap to keep safe: `--skip-existing` skips folder documents whose
output already exists, and Zotero runs skip items that already have a summary
note unless you pass `--force`. A replaced Zotero note is deleted only after
the new one is saved, so a failed run never loses an existing summary. The
tool also ignores its own previous output when walking a folder.

## Notes on tuning

Chunk size is derived from the context the summary model is **loaded** with,
which the tool prints when it is much smaller than the model supports:

```
note: google/gemma-4-26b-a4b-qat is loaded with 8192 of 262144 possible
context tokens; raising it in LM Studio would mean fewer chunks and a
better summary.
```

Raising the context in LM Studio's model load settings is the single biggest
quality win for long documents: fewer chunks means less information lost in
the map-reduce step. `--chunk-chars` overrides the derived value if you want
to set it yourself.

OCR runs at roughly 5–15 s per page, so a long scan takes a while; the tool
reports progress per page and elapsed time. `--max-minutes` bounds a Zotero
batch — it stops *starting* new items, always letting the one in flight
finish, and a rerun continues where it left off.

## Files

| File | Purpose |
| --- | --- |
| `sumocr.py` | CLI: sources, output paths, run control |
| `lmstudio.py` | LM Studio client, context sizing, OCR calls |
| `extract.py` | Per-page text-layer detection, extraction, OCR stitching |
| `summarize.py` | Prompts, language handling, map-reduce |
| `render.py` | LaTeX→Unicode, Markdown/HTML/text output |
| `zotero_source.py` | Zotero items, collections, attachments, notes |
| `config.py` | `config.ini` and environment settings |
| `test_units.py` | Checks for the pure functions — no network, no models |

```bash
python test_units.py
```

## License and attribution

Apache License 2.0 — see [LICENSE](LICENSE).

This continues my two earlier Apache-2.0 projects,
[doc-ollama-summarizer][doc] and [zotero-ollama-summarizer][zot]; all three
were written together with [Claude Code][cc]. [NOTICE](NOTICE) records which
code carried over and what changed.

[cc]: https://claude.com/claude-code
