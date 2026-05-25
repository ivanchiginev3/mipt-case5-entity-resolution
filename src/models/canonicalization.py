"""
Canonical entity construction.

Input:
clustered records with entity_id

Output:
one row per entity.
"""

import pandas as pd


def most_frequent_value(series: pd.Series):
    series = series.dropna()

    if len(series) == 0:
        return None

    return series.value_counts().index[0]


def collect_unique_values(series: pd.Series, max_values: int = 20) -> list:
    values = (
        series
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return values[:max_values]


def build_canonical_entities(
    clustered_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Формирует таблицу уникальных сущностей.
    """

    entities = (
        clustered_df
        .groupby("entity_id")
        .agg(
            canonical_name=("canonical_name", most_frequent_value),
            cluster_size=("record_id", "size"),
            canonical_country=("country", most_frequent_value),
            canonical_script=("script", most_frequent_value),
            main_relation_kind=("relation_kind", most_frequent_value),
            main_relation_role=("relation_role", most_frequent_value),
            unique_names_count=("name_latin", "nunique"),
            unique_countries_count=("country", "nunique"),
            unique_companies_count=("company_public_id", "nunique"),
            record_ids=("record_id", collect_unique_values),
            original_names=("party_name", collect_unique_values),
            normalized_names=("name_latin", collect_unique_values),
            countries=("country", collect_unique_values),
            companies=("company_public_id", collect_unique_values),
        )
        .reset_index()
        .sort_values("cluster_size", ascending=False)
    )

    return entities


def add_entity_quality_flags(
    entities: pd.DataFrame,
    large_cluster_threshold: int = 20,
    max_unique_names: int = 5,
    max_unique_countries: int = 3,
) -> pd.DataFrame:
    """
    Добавляет флаги качества entity-кластера.
    """

    entities = entities.copy()

    entities["is_singleton"] = (
        entities["cluster_size"] == 1
    ).astype(int)

    entities["is_large_cluster"] = (
        entities["cluster_size"] >= large_cluster_threshold
    ).astype(int)

    entities["has_many_names"] = (
        entities["unique_names_count"] > max_unique_names
    ).astype(int)

    entities["has_many_countries"] = (
        entities["unique_countries_count"] > max_unique_countries
    ).astype(int)

    entities["is_suspicious_entity"] = (
        (entities["has_many_names"] == 1)
        |
        (entities["has_many_countries"] == 1)
    ).astype(int)

    return entities


def entity_summary(entities: pd.DataFrame) -> None:
    print("=" * 60)
    print("CANONICAL ENTITY SUMMARY")
    print("=" * 60)

    print(f"Entities: {len(entities):,}")

    print("\nCluster size stats:")
    print(entities["cluster_size"].describe())

    print("\nSingletons:")
    print(entities["is_singleton"].value_counts())

    print("\nLarge clusters:")
    print(entities["is_large_cluster"].value_counts())

    print("\nSuspicious entities:")
    print(entities["is_suspicious_entity"].value_counts())