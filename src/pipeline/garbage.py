"""
Garbage entity detection.

Используется в:
1. dataset cleaning;
2. feature engineering;
3. inference post-filtering.
"""

import re
import pandas as pd


GARBAGE_PATTERNS = [
    r"замените\s+на",
    r"согласно\s+спис",
    r"согласно\s+по\s+спис",
    r"согласно\s+представ",
    r"согласно\s+предостав",
    r"представленного\s+спис",
    r"предоставленного\s+спис",
    r"unknown",
    r"неизвест",
    r"анхай",
    r"anhayt",
    r"kazmakerpoutyoun",
]


def is_garbage_entity(text) -> int:
    """
    Возвращает 1, если строка похожа на служебную/мусорную сущность.
    """

    if pd.isna(text):
        return 1

    text = str(text).lower().strip()

    if not text:
        return 1

    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, text):
            return 1

    return 0