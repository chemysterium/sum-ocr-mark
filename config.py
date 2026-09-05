"""Settings from config.ini, overridable by environment variables.

Resolution order for every setting: environment variable (the key, uppercased)
first, then config.ini next to this file, then the built-in default.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

import lmstudio

CONFIG_PATH = Path(__file__).resolve().parent / "config.ini"

_CONFIG = configparser.ConfigParser()
if CONFIG_PATH.exists():
    _CONFIG.read(CONFIG_PATH, encoding="utf-8")


def setting(section: str, key: str, default: str = "") -> str:
    env_value = os.environ.get(key.upper())
    if env_value:
        return env_value
    return _CONFIG.get(section, key, fallback=default)


LMSTUDIO_URL = setting("lmstudio", "lmstudio_url", lmstudio.DEFAULT_URL)
CHAT_MODEL = setting("lmstudio", "chat_model", lmstudio.DEFAULT_CHAT_MODEL)
OCR_MODEL = setting("lmstudio", "ocr_model", lmstudio.DEFAULT_OCR_MODEL)

ZOTERO_LIBRARY_ID = setting("zotero", "zotero_library_id")
ZOTERO_LIBRARY_TYPE = setting("zotero", "zotero_library_type", "user")
ZOTERO_API_KEY = setting("zotero", "zotero_api_key")
ZOTERO_STORAGE_DIR = setting(
    "zotero", "zotero_storage_dir", str(Path.home() / "Zotero" / "storage")
)
