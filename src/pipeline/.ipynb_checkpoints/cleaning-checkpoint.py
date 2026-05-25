"""
Очистка данных
"""

import pandas as pd
from src.pipeline.garbage import is_garbage_entity

from src.config import (
    GARBAGE_PATTERNS,
    MIN_NAME_LENGTH,
    VALID_SCRIPTS
)

from src.utils import (
    detect_script,
    safe_lower
)


def remove_garbage_rows(df):
    """
    Удаляет строки с техническими и служебными сущностями.
    """

    print("=" * 60)
    print("REMOVE GARBAGE ROWS")
    print("=" * 60)

    initial_size = len(df)

    garbage_mask = (
        df["party_name"]
        .apply(is_garbage_entity)
        .astype(bool)
    )

    print(f"Garbage rows found: {garbage_mask.sum():,}")

    df = df[~garbage_mask].copy()

    print(f"Removed total: {initial_size - len(df):,}")

    return df


def add_script_column(df):

    print("="*60)
    print("SCRIPT DETECTION")
    print("="*60)

    df['script'] = df['party_name'].apply(detect_script)

    print(df['script'].value_counts())

    return df


def remove_invalid_scripts(df):

    initial_size = len(df)

    df = df[
        df['script'].isin(VALID_SCRIPTS)
    ].copy()

    print(f"Removed invalid scripts: {initial_size - len(df):,}")

    return df


def remove_empty_names(df):

    print("="*60)
    print("REMOVE EMPTY NAMES")
    print("="*60)

    initial_size = len(df)

    df['party_name'] = (
        df['party_name']
        .astype(str)
        .str.strip()
    )

    mask = (
        df['party_name'].str.len() >= MIN_NAME_LENGTH
    )

    df = df[mask].copy()

    print(f"Removed empty/short names: {initial_size - len(df):,}")

    return df


def parse_dates(df):

    print("="*60)
    print("DATE PARSING")
    print("="*60)

    df['snapshot_date'] = pd.to_datetime(
        df['source_snapshot_date'],
        errors='coerce'
    )

    print(
        "Parsed dates:",
        df['snapshot_date'].notna().sum()
    )

    return df