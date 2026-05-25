"""
Utilities for cleaning and splitting labeled pair datasets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def make_pair_key(
    pairs: pd.DataFrame,
    idx1_col: str = "idx1",
    idx2_col: str = "idx2",
) -> pd.DataFrame:
    """
    Создает симметричный ключ пары.
    Пары A-B и B-A считаются одной и той же парой.
    """

    pairs = pairs.copy()

    pairs["pair_left"] = pairs[[idx1_col, idx2_col]].min(axis=1)
    pairs["pair_right"] = pairs[[idx1_col, idx2_col]].max(axis=1)

    pairs["pair_key"] = (
        pairs["pair_left"].astype(str)
        + "_"
        + pairs["pair_right"].astype(str)
    )

    return pairs


def remove_duplicate_pairs(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Удаляет повторяющиеся пары.
    """

    pairs = make_pair_key(pairs)

    before = len(pairs)

    pairs = (
        pairs
        .drop_duplicates(subset=["pair_key"])
        .copy()
    )

    after = len(pairs)

    print(f"Pairs before deduplication: {before:,}")
    print(f"Pairs after deduplication:  {after:,}")
    print(f"Removed: {before - after:,}")

    return pairs


def get_entity_group_for_pair(
    pairs: pd.DataFrame,
    id_col_1: str = "party_public_id_1",
    id_col_2: str = "party_public_id_2",
) -> pd.Series:
    """
    Создает group_id для разделения train/valid/test.

    Для positive pair оба ID одинаковые.
    Для negative pair берем пару ID как группу.
    """

    id1 = pairs[id_col_1].astype(str)
    id2 = pairs[id_col_2].astype(str)

    left = id1.where(id1 <= id2, id2)
    right = id2.where(id1 <= id2, id1)

    return left + "__" + right


def split_pairs_by_entity_group(
    pairs: pd.DataFrame,
    test_size: float = 0.15,
    valid_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Делит пары на train/valid/test по entity group.

    Это снижает риск утечки данных между выборками.
    """

    pairs = pairs.copy()

    pairs["entity_group"] = get_entity_group_for_pair(pairs)

    groups = pairs["entity_group"].drop_duplicates()

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=test_size + valid_size,
        random_state=random_state,
    )

    relative_valid_size = valid_size / (test_size + valid_size)

    valid_groups, test_groups = train_test_split(
        temp_groups,
        test_size=1 - relative_valid_size,
        random_state=random_state,
    )

    train_pairs = pairs[pairs["entity_group"].isin(train_groups)].copy()
    valid_pairs = pairs[pairs["entity_group"].isin(valid_groups)].copy()
    test_pairs = pairs[pairs["entity_group"].isin(test_groups)].copy()

    return train_pairs, valid_pairs, test_pairs


def print_pair_summary(
    pairs: pd.DataFrame,
    name: str = "pairs",
) -> None:
    """
    Печатает краткую статистику по датасету пар.
    """

    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    print(f"Rows: {len(pairs):,}")

    print("\nLabel distribution:")
    print(pairs["label"].value_counts())

    print("\nLabel ratio:")
    print((pairs["label"].value_counts(normalize=True) * 100).round(2))

    if "pair_type" in pairs.columns:
        print("\nPair types:")
        print(pairs["pair_type"].value_counts())