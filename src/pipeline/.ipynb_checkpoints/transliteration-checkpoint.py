"""Транслитерация"""
from unidecode import unidecode
import pandas as pd

def transliterate_to_latin(text):
    """Перевод в латиницу через unidecode"""
    if pd.isna(text) or not text:
        return ""
    return unidecode(str(text))