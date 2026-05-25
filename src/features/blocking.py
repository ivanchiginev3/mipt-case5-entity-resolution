"""
Blocking strategies for Entity Resolution.

Цель:
сократить число пар-кандидатов, которые нужно передавать в matching model.
"""

import pandas as pd
from itertools import combinations


def safe_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def first_token(text):
    tokens = safe_text(text).split()
    if not tokens:
        return ""
    return tokens[0]


def sorted_tokens_key(text):
    tokens = safe_text(text).split()
    return " ".join(sorted(tokens))


def prefix_key(text, n=4):
    text = safe_text(text).replace(" ", "")
    if not text:
        return "unknown"
    return text[:n]


def first_token_prefix_key(text, n=4):
    token = first_token(text)
    if not token:
        return "unknown"
    return token[:n]


def length_bucket(text, bucket_size=5):
    text = safe_text(text)
    return len(text) // bucket_size


def token_count_bucket(text):
    return len(safe_text(text).split())


def make_block_keys(df, name_col="name_latin"):
    """
    Создает несколько blocking keys.
    """

    df = df.copy()

    df["block_prefix_4"] = (
        df[name_col]
        .apply(lambda x: prefix_key(x, n=4))
    )

    df["block_first_token_4"] = (
        df[name_col]
        .apply(lambda x: first_token_prefix_key(x, n=4))
    )

    df["block_prefix_len"] = (
        df[name_col].apply(lambda x: prefix_key(x, n=4))
        + "_"
        + df[name_col].apply(lambda x: str(length_bucket(x)))
    )

    df["block_first_token_len"] = (
        df[name_col].apply(lambda x: first_token_prefix_key(x, n=4))
        + "_"
        + df[name_col].apply(lambda x: str(length_bucket(x)))
    )

    df["block_sorted_tokens"] = (
        df[name_col]
        .apply(sorted_tokens_key)
    )

    return df


def block_size_stats(df, block_col):
    """
    Статистика размеров блоков.
    """

    stats = (
        df.groupby(block_col)
        .size()
        .reset_index(name="block_size")
        .sort_values("block_size", ascending=False)
    )

    return stats


def calculate_reduction_ratio(df, block_col):
    """
    Считает reduction ratio:
    насколько blocking сокращает число пар.
    """

    n = len(df)

    total_pairs = n * (n - 1) / 2

    block_sizes = (
        df.groupby(block_col)
        .size()
        .values
    )

    candidate_pairs = sum(
        size * (size - 1) / 2
        for size in block_sizes
    )

    reduction_ratio = 1 - candidate_pairs / total_pairs

    return {
        "total_pairs": total_pairs,
        "candidate_pairs": candidate_pairs,
        "reduction_ratio": reduction_ratio,
    }


def generate_candidate_pairs_from_block(
    df,
    block_col,
    max_block_size=500,
):
    """
    Генерирует пары-кандидаты внутри блоков.

    Очень большие блоки пропускаются, чтобы не взорвать память.
    """

    rows = []

    for block_value, group in df.groupby(block_col):
        if len(group) < 2:
            continue

        if len(group) > max_block_size:
            continue

        indices = group.index.tolist()

        for idx1, idx2 in combinations(indices, 2):
            rows.append(
                {
                    "idx1": idx1,
                    "idx2": idx2,
                    "block_col": block_col,
                    "block_value": block_value,
                }
            )

    return pd.DataFrame(rows)


def evaluate_blocking_recall(
    labeled_pairs,
    candidate_pairs,
):
    """
    Проверяет, какая доля positive pairs покрывается blocking.

    Это главная метрика blocking.
    """

    labeled = labeled_pairs.copy()
    candidates = candidate_pairs.copy()

    labeled["left"] = labeled[["idx1", "idx2"]].min(axis=1)
    labeled["right"] = labeled[["idx1", "idx2"]].max(axis=1)
    labeled["pair_key"] = labeled["left"].astype(str) + "_" + labeled["right"].astype(str)

    candidates["left"] = candidates[["idx1", "idx2"]].min(axis=1)
    candidates["right"] = candidates[["idx1", "idx2"]].max(axis=1)
    candidates["pair_key"] = candidates["left"].astype(str) + "_" + candidates["right"].astype(str)

    positive_pairs = set(
        labeled[labeled["label"] == 1]["pair_key"]
    )

    candidate_pair_keys = set(
        candidates["pair_key"]
    )

    covered = positive_pairs & candidate_pair_keys

    if len(positive_pairs) == 0:
        recall = 0.0
    else:
        recall = len(covered) / len(positive_pairs)

    return {
        "positive_pairs": len(positive_pairs),
        "covered_positive_pairs": len(covered),
        "blocking_recall": recall,
    }