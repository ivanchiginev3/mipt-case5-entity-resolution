"""
Validation utilities for party_public_id labels.

Задача файла:
1. Проверить, насколько party_public_id пригоден как разметка.
2. Посчитать размеры ID-кластеров.
3. Посчитать количество уникальных имен внутри ID.
4. Оценить похожесть имен внутри ID.
5. Разделить ID на надежные и подозрительные.
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz


def get_public_id_stats(
    df: pd.DataFrame,
    id_col: str = "party_public_id",
    name_col: str = "name_latin",
    country_col: str = "country",
    script_col: str = "script",
) -> pd.DataFrame:
    """
    Собирает базовую статистику по party_public_id.
    """

    labeled = df[df[id_col].notna()].copy()

    stats = (
        labeled
        .groupby(id_col)
        .agg(
            cluster_size=(name_col, "size"),
            unique_names=(name_col, "nunique"),
            countries=(country_col, "nunique"),
            scripts=(script_col, "nunique"),
        )
        .reset_index()
    )

    stats["unique_name_ratio"] = (
        stats["unique_names"] / stats["cluster_size"]
    )

    return stats


def estimate_cluster_similarity(
    df: pd.DataFrame,
    id_col: str = "party_public_id",
    name_col: str = "name_latin",
    max_names_per_cluster: int = 20,
) -> pd.DataFrame:
    """
    Оценивает похожесть имен внутри каждого party_public_id.

    Важно:
    низкая похожесть не всегда означает ошибку.
    Например, одно и то же название может быть на разных языках.
    Поэтому similarity используем для анализа, но не как жесткое правило.
    """

    rows = []

    labeled = df[df[id_col].notna()].copy()

    for public_id, group in labeled.groupby(id_col):
        names = (
            group[name_col]
            .dropna()
            .drop_duplicates()
            .head(max_names_per_cluster)
            .tolist()
        )

        if len(names) <= 1:
            avg_similarity = 100.0
            min_similarity = 100.0
        else:
            scores = []

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    scores.append(
                        fuzz.token_sort_ratio(names[i], names[j])
                    )

            avg_similarity = float(np.mean(scores))
            min_similarity = float(np.min(scores))

        rows.append(
            {
                id_col: public_id,
                "avg_name_similarity": avg_similarity,
                "min_name_similarity": min_similarity,
            }
        )

    return pd.DataFrame(rows)


def mark_trusted_public_ids(
    stats: pd.DataFrame,
    id_col: str = "party_public_id",
    max_cluster_size: int = 20_000,
    max_unique_names: int = 3,
    max_countries: int = 3,
    max_scripts: int = 3,
) -> pd.DataFrame:
    """
    Делит party_public_id на надежные и подозрительные.

    Важно:
    - одиночные ID не считаем шумом;
    - низкую строковую похожесть не используем как жесткий фильтр;
    - основной критерий: внутри ID не должно быть слишком много разных имен,
      стран и алфавитов.
    """

    stats = stats.copy()

    conditions = (
        (stats["cluster_size"] <= max_cluster_size)
        &
        (stats["unique_names"] <= max_unique_names)
        &
        (stats["countries"] <= max_countries)
        &
        (stats["scripts"] <= max_scripts)
    )

    stats["is_trusted"] = conditions

    return stats


def get_suspicious_public_ids(
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Возвращает подозрительные public_id.
    """

    return (
        stats[~stats["is_trusted"]]
        .sort_values(
            ["cluster_size", "unique_names"],
            ascending=False
        )
    )


def get_cross_language_like_clusters(
    stats: pd.DataFrame,
    similarity_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Находит кластеры, где имена сильно отличаются строково.

    Это не обязательно ошибка.
    Часто это переводы или альтернативные написания.
    """

    return (
        stats[
            stats["avg_name_similarity"] < similarity_threshold
        ]
        .sort_values("avg_name_similarity")
    )


def print_public_id_summary(
    stats: pd.DataFrame,
) -> None:
    """
    Печатает краткий отчет по party_public_id.
    """

    print("=" * 60)
    print("PUBLIC ID VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total public IDs: {len(stats):,}")

    if "is_trusted" in stats.columns:
        trusted = stats["is_trusted"].sum()
        noisy = len(stats) - trusted

        print(f"Trusted IDs: {trusted:,}")
        print(f"Noisy IDs: {noisy:,}")
        print(f"Trusted ratio: {trusted / len(stats) * 100:.2f}%")

    print("\nCluster size stats:")
    print(stats["cluster_size"].describe())

    print("\nUnique names stats:")
    print(stats["unique_names"].describe())

    if "avg_name_similarity" in stats.columns:
        print("\nAverage name similarity stats:")
        print(stats["avg_name_similarity"].describe())