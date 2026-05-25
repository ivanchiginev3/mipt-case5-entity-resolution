"""
Production-grade normalization module for Entity Resolution

Design goals:
- NO information loss
- reversible transformations where possible
- separation of concerns (structural vs semantic cleaning)
- safe for ML features, blocking, embeddings
"""

import re
import unicodedata
from typing import Optional


# -----------------------------------------------------
# LEVEL 1 — SAFE NORMALIZATION (NO LOSS)
# -----------------------------------------------------

def normalize_unicode(text: Optional[str]) -> str:
    """Unicode normalization only (no semantic changes)."""
    if not text:
        return ""

    return unicodedata.normalize("NFKC", str(text))


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace only."""
    return re.sub(r"\s+", " ", text).strip()


def safe_lower(text: str) -> str:
    """Lowercase without any other transformation."""
    return str(text).lower() if text else ""


# -----------------------------------------------------
# LEVEL 2 — STRUCTURAL NORMALIZATION
# (still NO loss of semantic content)
# -----------------------------------------------------

APOSTROPHES = ["’", "`", "´", "‘"]
QUOTES = ["\"", "'", "«", "»"]


def normalize_quotes(text: str) -> str:
    for q in APOSTROPHES + QUOTES:
        text = text.replace(q, "'")
    return text


def normalize_dashes(text: str) -> str:
    """Unify different dash types."""
    return re.sub(r"[–—−]", "-", text)


def normalize_separators(text: str) -> str:
    """Normalize separators but DO NOT remove them."""
    text = re.sub(r"[|/\\_]+", " ", text)
    return text


def normalize_punctuation_soft(text: str) -> str:
    """
    Soft punctuation normalization:
    - replaces noise punctuation with space
    - preserves meaningful symbols like & and -
    """
    text = re.sub(r"[\.,;:\(\)\[\]\{\}]", " ", text)
    return text


def normalize_digits_keep(text: str) -> str:
    """
    IMPORTANT:
    We DO NOT remove digits.
    They are often semantically meaningful (branches, models, etc.).
    """
    return text


"""
Нормализация текста
"""

import re
import unicodedata
import pandas as pd

from src.config import LEGAL_FORMS


def normalize_unicode(text):

    if pd.isna(text):
        return ''

    return unicodedata.normalize('NFKD', str(text))


def remove_punctuation(text):

    text = re.sub(r'[^\w\s]', ' ', text)

    return text


def remove_extra_spaces(text):

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def lowercase(text):

    return str(text).lower()


def remove_digits(text):

    return re.sub(r'\d+', ' ', text)


def normalize_text(text):

    if pd.isna(text):
        return ''

    text = normalize_unicode(text)

    text = lowercase(text)

    text = remove_punctuation(text)

    text = remove_digits(text)

    text = remove_extra_spaces(text)

    return text


def remove_legal_forms(text):

    if not text:
        return ''

    tokens = text.split()

    tokens = [
        t for t in tokens
        if t not in LEGAL_FORMS
    ]

    return ' '.join(tokens)
