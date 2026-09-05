"""Prompts, language handling and map-reduce summarization.

Chunk sizes are not a constant here: they come from the context the summary
model was actually loaded with in LM Studio, because that is the number that
decides whether a chunk fits.
"""

from __future__ import annotations

from lmstudio import LMStudio, log

# How much of a chunk boundary to repeat in the next chunk, so a sentence cut
# in half is still seen whole once.
CHUNK_OVERLAP = 500

STYLE_PROMPTS = {
    "general": (
        "Write a structured summary of the following document. Cover its purpose, "
        "the main points in order, and any conclusions, decisions or "
        "recommendations it reaches."
    ),
    "paper": (
        "Write a structured summary (background, methods, key findings, "
        "limitations) of the following paper."
    ),
}

SYSTEM_PROMPT = (
    "You are a precise research assistant. Summarize documents accurately, "
    "preserving key findings, methods, and limitations. Do not invent information "
    "not present in the text. Write in plain Markdown. Never use LaTeX notation "
    "($...$, \\text{}, ^{} etc.); write formulas, isotopes and math with plain "
    "Unicode characters instead (e.g. H₂O, ⁶Li/⁷Li, 10⁻³, ≈, °C). Begin directly "
    'with the summary itself — no introductory sentence such as "Here is the '
    'summary".'
)

# Text that came from OCR carries errors the model should not treat as data.
OCR_CAVEAT = (
    " Part of this text was produced by OCR from scanned pages, so it may contain "
    "recognition errors, broken words and garbled numbers. Read through obvious "
    "OCR noise, and do not report a garbled value as if it were a real result."
)

# Shortcuts for --language. Any other value is passed through as a language
# name, so --language Polish or --language "Brazilian Portuguese" work too.
LANGUAGE_NAMES = {
    "auto": None,
    "en": "English", "eng": "English", "english": "English",
    "cs": "Czech", "cz": "Czech", "czech": "Czech", "cesky": "Czech",
    "sk": "Slovak", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pl": "Polish", "ru": "Russian", "uk": "Ukrainian",
}

# A directive written in the target language itself is a much stronger cue for
# which language to answer in than an English sentence naming it. Languages
# absent here fall back to the English instruction alone.
LANGUAGE_DIRECTIVES = {
    "Czech": "Napiš celé shrnutí v češtině.",
    "Slovak": "Napíš celé zhrnutie v slovenčine.",
    "German": "Schreibe die gesamte Zusammenfassung auf Deutsch.",
    "Polish": "Napisz całe podsumowanie w języku polskim.",
    "French": "Rédige l'intégralité du résumé en français.",
    "Spanish": "Escribe todo el resumen en español.",
    "Italian": "Scrivi l'intero riassunto in italiano.",
    "English": "Write the entire summary in English.",
}


def resolve_language(value: str) -> str | None:
    """Map a --language value to a language name; None means match the document."""
    key = value.strip().lower()
    if key in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[key]
    return value.strip()


def language_reminder(language: str | None) -> str:
    """The language rule, repeated at the end of every prompt.

    The system prompt alone is not enough: every task instruction is in
    English, and on a long document a model tends to follow the language of
    the instructions nearest its output rather than a system message far
    above it. Restating the rule after the document text keeps it in view.
    """
    if not language:
        return (
            "IMPORTANT: write your answer in the same language as the document text "
            "above, not in English, unless the document itself is English."
        )
    native = LANGUAGE_DIRECTIVES.get(language)
    english = f"IMPORTANT: write your entire answer in {language}."
    return f"{english} {native}" if native else english


def build_system_prompt(language: str | None, from_ocr: bool = False) -> str:
    """Add the language rule, and an OCR caveat when the text was scanned.

    Without a language rule a model answers in the language of the
    instructions — which are English — so a Czech document would come back
    summarized in English.
    """
    if language:
        rule = (
            f"Always write your summary in {language}, whatever language the document "
            "itself is written in."
        )
    else:
        rule = (
            "Write your summary in the same language as the document itself: a Czech "
            "document must be summarized in Czech, an English document in English, "
            "and so on. Keep established technical terms and proper names in their "
            "original form."
        )
    return SYSTEM_PROMPT + (OCR_CAVEAT if from_ocr else "") + " " + rule


def chunk_text(text: str, chunk_chars: int, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    if overlap >= chunk_chars:
        overlap = chunk_chars // 10

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def summarize(
    client: LMStudio,
    text: str,
    style: str,
    model: str,
    language: str | None = None,
    chunk_chars: int | None = None,
    max_tokens: int = 2048,
    from_ocr: bool = False,
) -> str:
    """Summarize `text`, in one pass or map-reduce if it does not fit."""
    instruction = STYLE_PROMPTS[style]
    system = build_system_prompt(language, from_ocr)
    # Repeated after the text of every prompt, where it stays closest to the
    # model's output — see language_reminder().
    reminder = language_reminder(language)

    if chunk_chars is None:
        chunk_chars = client.context_chars(model)
    chunks = chunk_text(text, chunk_chars)

    if len(chunks) == 1:
        return client.chat(
            system, f"{instruction}\n\n{chunks[0]}\n\n{reminder}", model, max_tokens
        )

    log(f"  document needs {len(chunks)} chunks of up to {chunk_chars} chars")
    partials: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        log(f"  summarizing part {i}/{len(chunks)}...")
        partials.append(
            client.chat(
                system,
                f"This is part {i} of {len(chunks)} of a longer document. Summarize "
                f"the key points in this excerpt:\n\n{chunk}\n\n{reminder}",
                model,
                max_tokens,
            )
        )

    log("  combining partial summaries...")
    combined = "\n\n".join(partials)
    # The partials are themselves a document, and on a small context window
    # they can add up to more than fits. Fold them down before the final pass.
    while len(combined) > chunk_chars:
        log(f"  partial summaries total {len(combined)} chars; condensing...")
        groups = chunk_text(combined, chunk_chars)
        combined = "\n\n".join(
            client.chat(
                system,
                "Condense these partial summaries of one document into a shorter set "
                f"of notes, losing no key facts:\n\n{group}\n\n{reminder}",
                model,
                max_tokens,
            )
            for group in groups
        )

    return client.chat(
        system,
        "Below are partial summaries of consecutive sections of one document. "
        f"Combine them into a single coherent summary. {instruction}\n\n"
        f"{combined}\n\n{reminder}",
        model,
        max_tokens,
    )
