"""Turn a model's Markdown answer into the requested output format.

Local models are told not to emit LaTeX and do it anyway, and DeepSeek-OCR
emits \\( ... \\) around subscripts even when asked for plain text, so math
spans are normalised to Unicode here rather than trusted to the prompt.
"""

from __future__ import annotations

import html as html_module
import re

_LATEX_SYMBOLS = {
    r"\approx": "≈", r"\sim": "~", r"\times": "×", r"\pm": "±", r"\cdot": "·",
    r"\degree": "°", r"\rightarrow": "→", r"\to": "→", r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥", r"\infty": "∞", r"\%": "%",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\Delta": "Δ",
    r"\epsilon": "ε", r"\lambda": "λ", r"\mu": "μ", r"\pi": "π", r"\sigma": "σ",
    r"\tau": "τ", r"\phi": "φ", r"\omega": "ω",
}

_SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "−": "⁻", "=": "⁼",
    "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ",
}

_SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "−": "₋", "=": "₌",
    "(": "₍", ")": "₎", "a": "ₐ", "e": "ₑ", "o": "ₒ", "x": "ₓ", "h": "ₕ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "p": "ₚ", "s": "ₛ", "t": "ₜ",
    "i": "ᵢ", "j": "ⱼ", "r": "ᵣ", "u": "ᵤ", "v": "ᵥ",
}

# A math span never starts or ends with whitespace, and never contains < or >.
# Both rules live in the pattern so money-like "$5 and $10" simply does not
# match: were it matched then declined, re.sub would still consume its dollar
# signs and a real span right after it could no longer be found.
_MATH_INNER = r"[^$\n<>\s](?:[^$\n<>]*[^$\n<>\s])?"

# \( ... \) and \[ ... \] have no such ambiguity — nothing but math uses them —
# so they may span whitespace and are matched non-greedily across one block.
_PAREN_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)


def _script(inner: str, table: dict, tag: str, html: bool) -> str:
    """Render a sub/superscript, preferring Unicode over markup.

    Unicode keeps the summary readable as plain text; HTML tags are used only
    when the target is HTML and some character has no Unicode equivalent.
    """
    if inner and all(ch in table for ch in inner):
        return "".join(table[ch] for ch in inner)
    if html:
        return f"<{tag}>{inner}</{tag}>"
    return f"^{inner}" if tag == "sup" else f"_{inner}"


def _convert_math_span(expr: str, html: bool) -> str:
    expr = re.sub(r"\\(?:text|mathrm|mathit|mathbf)\{([^{}]*)\}", r"\1", expr)
    expr = expr.replace(r"^\circ", "°")
    for macro, char in _LATEX_SYMBOLS.items():
        expr = expr.replace(macro, char)
    expr = re.sub(
        r"\^\{([^{}]*)\}|\^(\S)",
        lambda m: _script(m.group(1) or m.group(2), _SUPERSCRIPTS, "sup", html),
        expr,
    )
    expr = re.sub(
        r"_\{([^{}]*)\}|_(\S)",
        lambda m: _script(m.group(1) or m.group(2), _SUBSCRIPTS, "sub", html),
        expr,
    )
    return expr.replace("{", "").replace("}", "").strip()


def strip_latex(text: str, html: bool = False) -> str:
    """Turn $...$, \\(...\\) and \\[...\\] math into readable text.

    Dollar spans with LaTeX-ish characters (_, ^ or \\) are converted; others
    just lose the dollar signs ($120$, $NaOH$). Ordinary text between dollar
    signs ("costs $5 and $10") is left alone.
    """
    def replace_dollar(match: re.Match) -> str:
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        if re.search(r"[_^\\]", inner):
            return _convert_math_span(inner, html)
        return inner

    def replace_paren(match: re.Match) -> str:
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        return _convert_math_span(inner.strip(), html)

    text = _PAREN_MATH.sub(replace_paren, text)
    text = re.sub(rf"\$\$({_MATH_INNER})\$\$|\$({_MATH_INNER})\$", replace_dollar, text)
    return _reattach_scripts(text)


# Characters a converted sub/superscript can consist of.
_SCRIPT_CHARS = "".join(sorted(set(_SUPERSCRIPTS.values()) | set(_SUBSCRIPTS.values())))


def _reattach_scripts(text: str) -> str:
    """Close the gap a converted math span leaves behind.

    OCR writes a subscript as its own span — "HF-HNO \\( _{3} \\) at 120 °C" —
    so converting the span in place would leave "HF-HNO ₃  at", with the
    subscript detached from the formula it belongs to.
    """
    text = re.sub(rf"(?<=[A-Za-z0-9])[ \t]+(?=[{_SCRIPT_CHARS}])", "", text)
    return re.sub(rf"(?<=[{_SCRIPT_CHARS}])[ \t]{{2,}}", " ", text)


_LIST_LINE = re.compile(r"^\s{0,3}(?:[*+-]\s+|\d+[.)]\s+)\S")


def blank_line_before_lists(md: str) -> str:
    """Insert a blank line between a text line and a list right after it.

    Markdown only starts a list when a blank line precedes it, but models
    routinely write "**Key findings**" immediately followed by "* item";
    without this those bullets render as literal asterisks.
    """
    out: list[str] = []
    for line in md.split("\n"):
        if _LIST_LINE.match(line) and out and out[-1].strip() and not _LIST_LINE.match(out[-1]):
            out.append("")
        out.append(line)
    return "\n".join(out)


def to_markdown(text: str) -> str:
    return strip_latex(blank_line_before_lists(text))


def to_plain_text(text: str) -> str:
    """Strip Markdown syntax for readable plain text."""
    text = strip_latex(text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s{0,3}[*+-]\s+", "• ", text, flags=re.MULTILINE)
    return text


def to_note_html(md: str) -> str:
    """Render Markdown to the HTML stored in a Zotero note.

    Zotero notes are HTML; the model answers in Markdown (mirroring the
    Markdown document text it was given), so this converts before saving or
    the note would show raw **bold**/### markup. nl2br keeps single newlines
    visible as line breaks, which models use to separate headings and bullets.
    """
    import markdown

    return markdown.markdown(
        strip_latex(blank_line_before_lists(md), html=True),
        extensions=["sane_lists", "tables", "nl2br"],
    )


def to_html(md: str, title: str) -> str:
    import markdown

    # The page already has an <h1> naming the document, so a top-level heading
    # from the model is demoted rather than producing a second <h1>.
    demoted = re.sub(r"(?m)^#(?=\s)", "##", md)
    body = markdown.markdown(
        strip_latex(blank_line_before_lists(demoted), html=True),
        extensions=["sane_lists", "tables", "nl2br"],
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_module.escape(title)}</title>
<style>
  body {{ max-width: 46rem; margin: 2rem auto; padding: 0 1rem;
         font: 16px/1.6 system-ui, sans-serif; color: #222; }}
  h1 {{ font-size: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: .3rem .6rem; text-align: left; }}
  code {{ background: #f4f4f4; padding: .1rem .3rem; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1b1b1b; color: #e6e6e6; }}
    h1 {{ border-color: #444; }}
    th, td {{ border-color: #555; }}
    code {{ background: #2a2a2a; }}
  }}
</style>
</head>
<body>
<h1>{html_module.escape(title)}</h1>
{body}
</body>
</html>
"""


def render(text: str, fmt: str, title: str) -> str:
    if fmt == "html":
        return to_html(text, title)
    if fmt == "txt":
        return to_plain_text(text)
    return to_markdown(text)
