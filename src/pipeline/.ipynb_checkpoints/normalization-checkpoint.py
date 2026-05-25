"""Нормализация текста"""
import re
import unicodedata
import pandas as pd

# Стоп-слова
STOPWORDS = {
    'ru': ['согласно', 'списку', 'список', 'списка', 'представленному',
           'по', 'просто', 'неизвестная', 'неизвестный', 'неизвестное',
           'организация', 'компания', 'учреждение', 'предприятие',
           'физическое', 'лицо', 'юридическое', 'замените'],
    'en': ['unknown', 'according', 'list', 'simply', 'replace',
           'organization', 'company', 'institution', 'enterprise',
           'individual', 'legal', 'entity', 'person'],
    'arm': ['անհայտ', 'կազմակերպություն', 'ընկերություն']
}

# Юридические формы
LEGAL_FORMS_MAP = {
    'ооо': 'ooo', 'зао': 'zao', 'оао': 'oao', 'пао': 'pao', 'ао': 'ao',
    'тоо': 'too', 'ип': 'ip', 'чп': 'chp', 'гп': 'gp',
    'llc': 'llc', 'ltd': 'ltd', 'inc': 'inc', 'corp': 'corp',
    'co': 'co', 'gmbh': 'gmbh', 'plc': 'plc',
    'спԸ': 'spy', 'փբԸ': 'pby', 'բԸ': 'by'
}

ALL_STOPWORDS = STOPWORDS['ru'] + STOPWORDS['en'] + STOPWORDS['arm']


def basic_clean(text):
    """Базовая очистка: lower + удаление спецсимволов"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[«»"\'.,;:!?@#$%^&*()\[\]{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def advanced_normalize(text):
    """Продвинутая нормализация"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'[«»"\'.,;:!?@#$%^&*()\[\]{}<>|/\\–—\-]', ' ', text)
    text = re.sub(r'\b[IVXLCDM]+\b', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    """Удаление стоп-слов"""
    if not text:
        return text
    tokens = [t for t in text.split() if t not in ALL_STOPWORDS]
    return ' '.join(tokens)


def normalize_legal_forms(text):
    """Замена юрформ на стандартные"""
    if not text:
        return text
    tokens = text.split()
    normalized = [LEGAL_FORMS_MAP.get(t, t) for t in tokens]
    return ' '.join(normalized)


def remove_legal_forms(text):
    """Удаление юрформ"""
    if not text:
        return text
    tokens = [t for t in text.split() if t not in LEGAL_FORMS_MAP]
    return ' '.join(tokens)