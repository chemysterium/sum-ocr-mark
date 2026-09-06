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
- **Scans become searchable.** `--text-layer` writes the OCR-ed words back
  into the PDF as invisible, positioned text, so Zotero can index a scan and
  you can search it in a reader. `--replace-pdf` does it to the attachment
  in place, keeping a backup.
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

For `--text-layer`, also install Tesseract — it writes the searchable
layer, and places words far more accurately than a vision model can:

- Windows: <https://github.com/UB-Mannheim/tesseract/wiki>
- Debian/Ubuntu: `apt install tesseract-ocr`
- macOS: `brew install tesseract`

Add the language packs you need (`tesseract-ocr-ces` for Czech). The tool
finds Tesseract even when the Windows installer leaves it off PATH.

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

# Make every scanned PDF in the library searchable, nothing else
python sumocr.py --zotero-all --zotero-local --action none \
    --text-layer --replace-pdf
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
extracted text with no LLM involved; `both` writes both files; `none`
writes neither, for runs whose only purpose is `--text-layer`.

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

### Making the PDF itself searchable

By default OCR is only used to *read* a scan — the PDF is left alone.
`--text-layer` writes the OCR-ed words back into the file as invisible text,
positioned over the pixels they came from, so the scan becomes searchable and
Zotero can index it. The page still looks exactly the same.

```bash
# Write a searchable copy next to the original, as <name>.ocr.pdf
python sumocr.py --file scan.pdf --text-layer

# Make the Zotero attachment itself searchable, in place
python sumocr.py --zotero-item ABCD1234 --text-layer --replace-pdf
```

To do only this — no summaries, no Markdown, just make scanned PDFs
searchable — use `--action none`:

```bash
# See which of your Zotero PDFs would be OCR-ed, changing nothing
python sumocr.py --zotero-all --zotero-local --action none --text-layer \
    --replace-pdf --dry-run

# Then do it
python sumocr.py --zotero-all --zotero-local --action none --text-layer \
    --replace-pdf
```

Such a run touches only what needs touching, and reports the rest as
skipped rather than failed:

| Case | What happens |
| --- | --- |
| Some pages lack a text layer | Only those pages are OCR-ed and written |
| Every page already has text | Skipped before anything is read |
| Item has no PDF attached | Skipped |
| PDF is not synced to this machine | Skipped |

```
[53/58] The Speciation of Fe(Ii) and Fe(Iii) in Natural-Waters (FQ4CZU7L)
  would skip: all 19 page(s) already have a text layer
[55/58] Hardening Mechanisms by Hexamethylenetetramine ... (VTRTE4V5)
  would skip: no PDF attachment
Done. 2 would process, 56 skipped, 0 failed.
```

Zotero answers every request in about the same two seconds whatever it
returns, so batch runs fetch all attachments in one paginated sweep rather
than asking per item, and list items 500 at a time instead of pyzotero's
default 25. Asking per item would make the request count, not the data,
decide the runtime.

A real library of 2842 items, on a laptop:

```
Listing every paper in the library...
Found 2842 papers in the library.
Fetching the attachment list...
  2045 PDF attachment(s) in 15s
...
Done. 93 would process, 2749 skipped, 0 failed.

real    2m49s
```

Of the 2749 skipped, 1942 were already searchable, 797 had no PDF and 10
were not synced locally. The 93 remaining files hold 5153 pages between
them but need only **402** pages OCR-ed, because most are partly scanned —
roughly 1.7 hours of model time, which `--max-minutes` can spread over
several sessions.

The already-searchable check reads page text only, and happens before the
Markdown extraction pass, so skipping a 100-page PDF is near-instant and
costs no model time. `--zotero-local` avoids needing an API key, and works
here because the PDFs are edited on disk rather than through the read-only
API.

`--replace-pdf` overwrites the original, keeping it as `<name>.pdf.bak` and
writing through a temporary file, so an interrupted run cannot leave a
truncated PDF where your attachment used to be. After replacing a Zotero
attachment, right-click the item and choose **Reindex Item** so Zotero picks
up the new text.

Pages that already have a real text layer are never touched, even under
`--ocr force`, so a page can't end up holding its own words twice.

#### Which engine writes the layer

Two OCR engines are used, for the two jobs each is good at:

| Output | Engine | Why |
| --- | --- | --- |
| PDF text layer | **Tesseract** (default) | Word-level geometry, so selecting and highlighting land on the right words |
| Markdown, summaries | **DeepSeek-OCR** | Better reading order and structure; coordinates are irrelevant here |

They are never reconciled, so there is no alignment step to go wrong.
Measured against the ink on a real scanned paper:

| Engine | Median word error | 90th pct | Worst |
| --- | --- | --- | --- |
| Tesseract | **0.33 pt** | 0.66 pt | 4.4 pt |
| DeepSeek blocks | 3.19 pt | 3.48 pt | 69.5 pt |

DeepSeek only reports *block* positions, and its text then has to be
re-flowed into those boxes in a substitute font, so individual words drift.
Tesseract reports every word's own box. It is also faster for this job:
16 s for a 7-page scan against 146 s.

`--text-layer-engine deepseek` selects the old behaviour if Tesseract is not
installed, and `--ocr-lang` picks Tesseract's language(s) — `--ocr-lang ces`
or `--ocr-lang eng+ces` for Czech, whichever packs you have.

A Tesseract text-layer run with `--action none` never contacts LM Studio at
all, so a library sweep needs no model loaded.

One more detail: the font for the DeepSeek engine is chosen by round-trip
test, not by name. PDFs record which glyph to draw plus a separate table
saying what each glyph means, and PyMuPDF builds that table by
reverse-mapping glyphs — with most fonts a plain hyphen comes back as U+2010
and a space as U+00A0, which silently breaks search for `AG50W-X12` or any
phrase. The tool writes a probe string with each candidate font and reads it
back, using the first that survives unchanged.

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
- `--output PATH` names one exact file; `--output-dir DIR` collects a batch,
  mirroring the source tree underneath it so that same-named files in
  different subfolders do not overwrite each other.
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

### Choosing models and limits

| Switch | Default | What it does |
| --- | --- | --- |
| `--model` | `google/gemma-4-26b-a4b-qat` | Model used for summaries |
| `--ocr-model` | `deepseek-ocr-2` | Vision model used for OCR |
| `--lmstudio-url` | `http://localhost:1234` | Where LM Studio is listening |
| `--chunk-chars` | from the loaded context | Characters per chunk |
| `--max-tokens` | `2048` | Cap on one summary response |
| `--timeout` | `900` | Seconds to wait for one response |

All three model settings can also live in `config.ini` (see
`config.example.ini`) or in the environment as `CHAT_MODEL`, `OCR_MODEL`
and `LMSTUDIO_URL`, so you need not repeat them on every run. Model names
are checked before any work starts, and a wrong one fails immediately with
the list of what LM Studio actually has loaded.

## Files

| File | Purpose |
| --- | --- |
| `sumocr.py` | CLI: sources, output paths, run control |
| `lmstudio.py` | LM Studio client, context sizing, OCR calls |
| `extract.py` | Per-page text-layer detection, extraction, OCR stitching |
| `summarize.py` | Prompts, language handling, map-reduce |
| `render.py` | LaTeX→Unicode, Markdown/HTML/text output |
| `textlayer.py` | Invisible text layers: box parsing, font choice, writing |
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
