"""
Основной preprocessing pipeline
"""

from src.pipeline.normalization import (
    normalize_text,
    remove_legal_forms
)

from src.pipeline.transliteration import (
    transliterate
)


def preprocess_names(df):

    print("="*60)
    print("NAME PREPROCESSING")
    print("="*60)

    # базовая нормализация
    df['name_normalized'] = (
        df['party_name']
        .apply(normalize_text)
    )

    # удаление юрформ
    df['name_no_legal'] = (
        df['name_normalized']
        .apply(remove_legal_forms)
    )

    # латинизация
    df['name_latin'] = (
        df['name_no_legal']
        .apply(transliterate)
    )

    print("Done.")

    return df