"""
Pair generation for Entity Resolution.
"""

import random
import pandas as pd
from itertools import combinations
from rapidfuzz import fuzz


def generate_positive_pairs(
    df: pd.DataFrame,
    trusted_ids: set,
    id_col: str = "party_public_id",
    max_pairs_per_id: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Генерирует positive pairs:
    пары записей с одинаковым надежным party_public_id.
    """

    random.seed(random_state)

    rows = []

    labeled = df[
        df[id_col].isin(trusted_ids)
    ].copy()

    for public_id, group in labeled.groupby(id_col):
        indices = group.index.tolist()

        if len(indices) < 2:
            continue

        all_pairs = list(combinations(indices, 2))

        if len(all_pairs) > max_pairs_per_id:
            all_pairs = random.sample(all_pairs, max_pairs_per_id)

        for idx1, idx2 in all_pairs:
            rows.append(
                {
                    "idx1": idx1,
                    "idx2": idx2,
                    "label": 1,
                    "pair_type": "positive_same_public_id",
                }
            )

    return pd.DataFrame(rows)


def generate_easy_negative_pairs(
    df: pd.DataFrame,
    trusted_ids: set,
    id_col: str = "party_public_id",
    n_pairs: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Генерирует easy negatives:
    случайные пары с разными party_public_id.
    """

    random.seed(random_state)

    labeled = df[
        df[id_col].isin(trusted_ids)
    ].copy()

    indices = labeled.index.tolist()

    rows = []

    attempts = 0
    max_attempts = n_pairs * 10

    while len(rows) < n_pairs and attempts < max_attempts:
        attempts += 1

        idx1, idx2 = random.sample(indices, 2)

        id1 = df.loc[idx1, id_col]
        id2 = df.loc[idx2, id_col]

        if id1 == id2:
            continue

        rows.append(
            {
                "idx1": idx1,
                "idx2": idx2,
                "label": 0,
                "pair_type": "easy_negative_different_public_id",
            }
        )

    return pd.DataFrame(rows)


def generate_hard_negative_pairs(
    df: pd.DataFrame,
    trusted_ids: set,
    id_col: str = "party_public_id",
    name_col: str = "name_latin",
    sample_size: int = 50_000,
    threshold: int = 80,
    max_pairs: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Генерирует hard negatives:
    имена похожи, но party_public_id разные.
    """

    random.seed(random_state)

    labeled = df[
        df[id_col].isin(trusted_ids)
    ].copy()

    if len(labeled) > sample_size:
        labeled = labeled.sample(sample_size, random_state=random_state)

    records = labeled[
        [id_col, name_col]
    ].reset_index()

    rows = []

    # простой blocking по первым 3 символам, чтобы не сравнивать всех со всеми
    records["simple_block"] = (
        records[name_col]
        .fillna("")
        .str[:3]
    )

    for _, block in records.groupby("simple_block"):
        if len(block) < 2:
            continue

        block_records = block.to_dict("records")

        for i in range(len(block_records)):
            for j in range(i + 1, len(block_records)):

                r1 = block_records[i]
                r2 = block_records[j]

                if r1[id_col] == r2[id_col]:
                    continue

                score = fuzz.token_sort_ratio(
                    r1[name_col],
                    r2[name_col]
                )

                if score >= threshold:
                    rows.append(
                        {
                            "idx1": r1["index"],
                            "idx2": r2["index"],
                            "label": 0,
                            "pair_type": "hard_negative_similar_name",
                            "name_similarity": score,
                        }
                    )

                if len(rows) >= max_pairs:
                    return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def attach_pair_columns(
    pairs: pd.DataFrame,
    df: pd.DataFrame,
    columns: list,
) -> pd.DataFrame:
    """
    Добавляет к парам признаки исходных записей:
    name_latin_1, name_latin_2, country_1, country_2 и т.д.
    """

    pairs = pairs.copy()

    left = df[columns].copy()
    right = df[columns].copy()

    left.columns = [f"{col}_1" for col in columns]
    right.columns = [f"{col}_2" for col in columns]

    pairs = pairs.join(left, on="idx1")
    pairs = pairs.join(right, on="idx2")

    return pairs