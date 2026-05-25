"""
Clustering utilities for Entity Resolution.

Pairwise duplicate predictions -> entity clusters.
"""

import pandas as pd
import networkx as nx


def build_duplicate_graph(
    predicted_pairs: pd.DataFrame,
    score_col: str = "duplicate_score",
    threshold: float = 0.5,
) -> nx.Graph:
    """
    Строит граф дублей.
    Узлы — индексы записей.
    Ребра — предсказанные дубликаты.
    """

    graph = nx.Graph()

    pairs = predicted_pairs[
        predicted_pairs[score_col] >= threshold
    ].copy()

    for _, row in pairs.iterrows():
        graph.add_edge(
            int(row["idx1"]),
            int(row["idx2"]),
            weight=float(row[score_col]),
        )

    return graph


def build_clusters_from_graph(
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Строит кластеры как connected components.
    """

    rows = []

    components = list(nx.connected_components(graph))

    for cluster_num, component in enumerate(components):
        entity_id = f"entity_{cluster_num:08d}"

        for idx in component:
            rows.append(
                {
                    "idx": idx,
                    "entity_id": entity_id,
                    "cluster_size": len(component),
                }
            )

    return pd.DataFrame(rows)


def attach_clusters_to_records(
    df: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет entity_id к исходным записям.

    Записи без дублей получают собственный single-record entity_id.
    """

    result = df.copy()

    result = result.reset_index().rename(columns={"index": "idx"})

    result = result.merge(
        clusters,
        on="idx",
        how="left",
    )

    missing_mask = result["entity_id"].isna()

    result.loc[missing_mask, "entity_id"] = (
        "single_"
        + result.loc[missing_mask, "idx"].astype(str)
    )

    result.loc[missing_mask, "cluster_size"] = 1

    result["cluster_size"] = result["cluster_size"].astype(int)

    return result


def compute_cluster_stats(
    clustered_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Считает статистику по итоговым entity clusters.
    """

    stats = (
        clustered_df
        .groupby("entity_id")
        .agg(
            cluster_size=("record_id", "size"),
            unique_names=("name_latin", "nunique"),
            unique_countries=("country", "nunique"),
            unique_scripts=("script", "nunique"),
            unique_relation_kinds=("relation_kind", "nunique"),
            unique_companies=("company_public_id", "nunique"),
        )
        .reset_index()
        .sort_values("cluster_size", ascending=False)
    )

    return stats


def get_large_clusters(
    cluster_stats: pd.DataFrame,
    min_size: int = 20,
) -> pd.DataFrame:
    """
    Возвращает крупные кластеры для ручной проверки.
    """

    return cluster_stats[
        cluster_stats["cluster_size"] >= min_size
    ].copy()


def get_suspicious_clusters(
    cluster_stats: pd.DataFrame,
    max_unique_names: int = 5,
    max_unique_countries: int = 3,
) -> pd.DataFrame:
    """
    Находит потенциально проблемные кластеры.
    """

    suspicious = cluster_stats[
        (cluster_stats["unique_names"] > max_unique_names)
        |
        (cluster_stats["unique_countries"] > max_unique_countries)
    ].copy()

    return suspicious.sort_values(
        ["cluster_size", "unique_names"],
        ascending=False,
    )


def choose_canonical_name(
    names: pd.Series,
) -> str:
    """
    Выбирает каноническое имя.

    Простая стратегия:
    берем самое частое нормализованное имя.
    """

    names = names.dropna()

    if len(names) == 0:
        return ""

    return names.value_counts().index[0]


def add_canonical_names(
    clustered_df: pd.DataFrame,
    name_col: str = "name_latin",
) -> pd.DataFrame:
    """
    Добавляет canonical_name для каждого entity_id.
    """

    canonical = (
        clustered_df
        .groupby("entity_id")[name_col]
        .apply(choose_canonical_name)
        .reset_index(name="canonical_name")
    )

    result = clustered_df.merge(
        canonical,
        on="entity_id",
        how="left",
    )

    return result

def filter_edges_for_clustering(
    scored_pairs: pd.DataFrame,
    score_threshold: float = 0.95,
    strict_score_threshold: float = 0.995,
) -> pd.DataFrame:
    """
    Фильтрует пары перед кластеризацией.

    Идея:
    - exact/core exact пары оставляем;
    - очень уверенные пары оставляем;
    - для перестановок токенов требуем высокий token_jaccard;
    - слабые цепочные связи убираем.
    """

    pairs = scored_pairs.copy()

    base = pairs["duplicate_score"] >= score_threshold

    exact = (
        (pairs["exact_match"] == 1)
        |
        (pairs["core_exact_match"] == 1)
    )

    very_confident = (
        pairs["duplicate_score"] >= strict_score_threshold
    )

    strong_token_overlap = (
        (pairs["token_jaccard"] >= 0.75)
        &
        (pairs["length_diff"] <= 5)
        &
        (pairs["core_token_set_ratio"] >= 95)
    )

    no_garbage = (
        pairs.get("has_garbage_entity", 0) == 0
    )

    filtered = pairs[
        base
        &
        no_garbage
        &
        (
            exact
            |
            very_confident
            |
            strong_token_overlap
        )
    ].copy()

    return filtered