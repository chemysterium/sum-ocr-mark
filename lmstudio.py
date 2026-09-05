"""Client for a local LM Studio server.

LM Studio exposes an OpenAI-compatible API on http://localhost:1234/v1 plus a
native /api/v0 that additionally reports, per model, how much context it was
actually loaded with. That number matters: a model whose weights support 262k
context is routinely *loaded* with 8k, and a chunk sized for the former
silently overflows the latter, so chunk sizes here are derived from the live
value rather than from a constant (see context_chars()).

Two models are used, both served by the same LM Studio instance:
  - a text model for summaries   (default: google/gemma-4-26b-a4b-qat)
  - a vision model for OCR       (default: deepseek-ocr-2)
"""

from __future__ import annotations

import base64
import sys

import requests

# Done here, in the module everything else imports, so that any entry point is
# safe: document titles and summaries routinely carry characters a legacy
# Windows console encoding (cp1250 and friends) cannot represent, and printing
# one would otherwise raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

DEFAULT_URL = "http://localhost:1234"
DEFAULT_CHAT_MODEL = "google/gemma-4-26b-a4b-qat"
DEFAULT_OCR_MODEL = "deepseek-ocr-2"

# DeepSeek-OCR understands several task prompts. "Free OCR." returns clean
# reading-order text; the <|grounding|> variant additionally emits bounding
# boxes as `label[[x, y, x, y]]` lines and LaTeX-wrapped subscripts, both of
# which are noise here — see clean_ocr_text() in extract.py for the cleanup
# that makes it usable if you do pick it.
OCR_PROMPT_FREE = "<image>\nFree OCR."
OCR_PROMPT_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown."
OCR_PROMPTS = {"free": OCR_PROMPT_FREE, "markdown": OCR_PROMPT_MARKDOWN}

# Room left for the model's own answer, and for the system prompt and chat
# template around it, when deriving a chunk size from the context window.
RESERVED_OUTPUT_TOKENS = 2048
RESERVED_OVERHEAD_TOKENS = 512

# Scientific prose tokenizes worse than plain English — digits, units, Greek
# letters and isotope notation each cost a token — so this sits deliberately
# below the usual ~4 chars/token rule of thumb.
CHARS_PER_TOKEN = 3.0


class LMStudioError(Exception):
    """A problem talking to LM Studio that the user needs to act on."""


def log(message: str) -> None:
    """Progress goes to stderr so stdout carries only the result."""
    print(message, file=sys.stderr)


class LMStudio:
    def __init__(self, url: str = DEFAULT_URL, timeout: int = 900) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        # Whether the server accepted reasoning_effort. Cleared after a
        # rejection so the fallback is taken once, not once per chunk.
        self._reasoning_ok = True
        self._models_cache: list[str] | None = None

    # ----------------------------------------------------------------- info

    def models(self) -> list[str]:
        if self._models_cache is None:
            try:
                resp = self.session.get(f"{self.url}/v1/models", timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise LMStudioError(self._unreachable(exc)) from None
            self._models_cache = [m["id"] for m in resp.json().get("data", [])]
        return self._models_cache

    def model_info(self, model: str) -> dict:
        """Per-model details from the native API, or {} if it is unavailable.

        /api/v0 exists only in recent LM Studio builds, so every caller has to
        cope with an empty dict rather than relying on this.
        """
        try:
            resp = self.session.get(f"{self.url}/api/v0/models", timeout=30)
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                if entry.get("id") == model:
                    return entry
        except (requests.RequestException, ValueError):
            pass
        return {}

    def check_model(self, model: str, role: str) -> None:
        """Fail early, with the list of available models, rather than mid-run."""
        available = self.models()
        if model not in available:
            raise LMStudioError(
                f"The {role} model '{model}' is not available in LM Studio.\n"
                f"Available models: {', '.join(available) or '(none)'}\n"
                "Load it in LM Studio's Developer tab, or pass a different name."
            )

    def context_chars(self, model: str, fallback: int = 16000) -> int:
        """Characters that safely fit in one request to `model`.

        Derived from the context the model was actually loaded with, minus room
        for the answer and the prompt scaffolding.
        """
        info = self.model_info(model)
        ctx = info.get("loaded_context_length") or info.get("max_context_length")
        if not ctx:
            return fallback

        usable = ctx - RESERVED_OUTPUT_TOKENS - RESERVED_OVERHEAD_TOKENS
        if usable < 1000:
            raise LMStudioError(
                f"Model '{model}' is loaded with only {ctx} tokens of context, which "
                "leaves no room for a document. Reload it in LM Studio with a larger "
                "context length."
            )

        maximum = info.get("max_context_length")
        if maximum and ctx < maximum / 2:
            log(
                f"  note: {model} is loaded with {ctx} of {maximum} possible context "
                "tokens; raising it in LM Studio would mean fewer chunks and a "
                "better summary."
            )
        return int(usable * CHARS_PER_TOKEN)

    # ----------------------------------------------------------------- chat

    def chat(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = RESERVED_OUTPUT_TOKENS,
        temperature: float = 0.3,
    ) -> str:
        """One completion, with the model's extended reasoning turned off.

        Thinking-capable models (gemma-4 among them) otherwise spend the whole
        output budget on hidden reasoning tokens and return an empty answer —
        the same failure the Ollama versions of these scripts avoided with
        "think": false. LM Studio's equivalent is reasoning_effort "none",
        which older builds reject, hence the fallback in _post().
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self._reasoning_ok:
            payload["reasoning_effort"] = "none"

        message = self._post(payload)
        content = (message.get("content") or "").strip()

        if not content and (message.get("reasoning_content") or "").strip():
            raise LMStudioError(
                f"'{model}' spent its entire {max_tokens}-token output budget on "
                "hidden reasoning and returned no answer. Raise --max-tokens, or "
                "turn thinking off for this model in LM Studio."
            )
        if not content:
            raise LMStudioError(f"'{model}' returned an empty response.")
        return content

    def ocr_image(
        self,
        png: bytes,
        model: str,
        prompt: str = OCR_PROMPT_FREE,
        max_tokens: int = 8192,
    ) -> str:
        """Transcribe one rendered page image."""
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        message = self._post(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            }
        )
        return (message.get("content") or "").strip()

    # -------------------------------------------------------------- private

    def _post(self, payload: dict) -> dict:
        try:
            resp = self.session.post(
                f"{self.url}/v1/chat/completions", json=payload, timeout=self.timeout
            )
        except requests.ConnectionError as exc:
            raise LMStudioError(self._unreachable(exc)) from None
        except requests.Timeout:
            raise LMStudioError(
                f"LM Studio did not answer within {self.timeout}s. A long document on "
                "a large model can need longer — raise --timeout."
            ) from None

        # Retry once without reasoning_effort if that is what the server
        # objected to, then remember not to send it again this run.
        if (
            resp.status_code == 400
            and "reasoning_effort" in payload
            and "reasoning" in resp.text.lower()
        ):
            self._reasoning_ok = False
            return self._post({k: v for k, v in payload.items() if k != "reasoning_effort"})

        if resp.status_code == 404:
            raise LMStudioError(
                f"LM Studio does not know the model '{payload['model']}'. "
                f"Available: {', '.join(self.models()) or '(none)'}"
            )
        if resp.status_code >= 400:
            raise LMStudioError(f"LM Studio returned {resp.status_code}: {resp.text[:500]}")

        try:
            return resp.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError):
            raise LMStudioError(f"Unexpected response from LM Studio: {resp.text[:500]}") from None

    def _unreachable(self, exc: Exception) -> str:
        return (
            f"Cannot reach LM Studio at {self.url} ({exc.__class__.__name__}).\n"
            "Start LM Studio, open the Developer tab and click 'Start Server', then "
            "load the models you want to use."
        )
