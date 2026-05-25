"""
Вспомогательные функции
"""

import re
import pandas as pd


def detect_script(text):

    if pd.isna(text):
        return 'unknown'

    text = str(text)

    has_cyr = bool(re.search(r'[А-Яа-яЁё]', text))
    has_lat = bool(re.search(r'[A-Za-z]', text))
    has_arm = bool(re.search(r'[\u0530-\u058F]', text))
    has_geo = bool(re.search(r'[\u10A0-\u10FF]', text))

    scripts = []

    if has_cyr:
        scripts.append('cyrillic')

    if has_lat:
        scripts.append('latin')

    if has_arm:
        scripts.append('armenian')

    if has_geo:
        scripts.append('georgian')

    if len(scripts) == 0:
        return 'unknown'

    if len(scripts) > 1:
        return 'mixed'

    return scripts[0]


def safe_lower(text):

    if pd.isna(text):
        return ''

    return str(text).lower().strip()