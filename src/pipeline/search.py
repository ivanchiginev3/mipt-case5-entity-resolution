"""
Search pipeline for Entity Resolution.

POST /search logic:
query record
→ normalization/transliteration
→ blocking
→ candidate retrieval
→ pair feature generation
→ model scoring
→ top-K results
"""

import pandas as pd

from src.pipeline.normalization import (
    normalize_text,
    remove_legal_forms,
)

from src.pipeline.transliteration import transliterate

from src.features.blocking import make_block_keys
from src.features.text_features import build_pair_features, get_feature_columns


def preprocess_query_record(record: dict) -> dict:
    """
    Приводит входящую запись к тому же виду,
    что и записи в processed dataset.
    """

    party_name = record.get("party_name", "")

    name_normalized = normalize_text(party_name)
    name_no_legal = remove_legal_forms(name_normalized)
    name_latin = transliterate(name_no_legal)

    processed = {
        "record_id": record.get("record_id", "query_record"),
        "party_name": party_name,
        "name_normalized": name_normalized,
        "name_no_legal": name_no_legal,
        "name_latin": name_latin,
        "country": record.get("country"),
        "script": record.get("script"),
        "relation_kind": record.get("relation_kind"),
        "relation_role": record.get("relation_role"),
        "company_public_id": record.get("company_public_id"),
        "company_name_norm": record.get("company_name_norm"),
        "party_public_id": record.get("party_public_id"),
    }

    return processed


def build_query_dataframe(record: dict) -> pd.DataFrame:
    """
    Создает DataFrame из одной query-записи и строит block keys.
    """

    processed = preprocess_query_record(record)

    query_df = pd.DataFrame([processed])
    query_df.index = [-1]

    query_df = make_block_keys(
        query_df,
        name_col="name_latin",
    )

    return query_df


def get_candidate_records(
    query_df: pd.DataFrame,
    database_df: pd.DataFrame,
    block_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Находит кандидатов из базы по нескольким blocking keys.
    """

    if block_cols is None:
        block_cols = [
            "block_prefix_len",
            "block_first_token_len",
            "block_sorted_tokens",
        ]

    candidate_indices = set()

    query_row = query_df.iloc[0]

    for block_col in block_cols:
        block_value = query_row[block_col]

        matched = database_df[
            database_df[block_col] == block_value
        ]

        candidate_indices.update(matched.index.tolist())

    candidates = database_df.loc[list(candidate_indices)].copy()

    return candidates


def build_query_candidate_pairs(
    query_df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Создает пары query-record ↔ candidate-records.
    """

    query_row = query_df.iloc[0]

    rows = []

    for idx, candidate in candidates.iterrows():
        rows.append(
            {
                "idx1": -1,
                "idx2": idx,

                "record_id_1": query_row.get("record_id"),
                "party_public_id_1": query_row.get("party_public_id"),
                "party_name_1": query_row.get("party_name"),
                "name_latin_1": query_row.get("name_latin"),
                "name_no_legal_1": query_row.get("name_no_legal"),
                "country_1": query_row.get("country"),
                "script_1": query_row.get("script"),
                "relation_kind_1": query_row.get("relation_kind"),
                "relation_role_1": query_row.get("relation_role"),
                "company_public_id_1": query_row.get("company_public_id"),
                "company_name_norm_1": query_row.get("company_name_norm"),

                "record_id_2": candidate.get("record_id"),
                "party_public_id_2": candidate.get("party_public_id"),
                "party_name_2": candidate.get("party_name"),
                "name_latin_2": candidate.get("name_latin"),
                "name_no_legal_2": candidate.get("name_no_legal"),
                "country_2": candidate.get("country"),
                "script_2": candidate.get("script"),
                "relation_kind_2": candidate.get("relation_kind"),
                "relation_role_2": candidate.get("relation_role"),
                "company_public_id_2": candidate.get("company_public_id"),
                "company_name_norm_2": candidate.get("company_name_norm"),
            }
        )

    return pd.DataFrame(rows)


def score_search_pairs(
    pairs: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Считает признаки и similarity score для query-candidate pairs.
    """

    if pairs.empty:
        return pairs

    scored = build_pair_features(pairs)

    feature_cols = get_feature_columns()

    leakage_like_features = [
        "same_company_public_id",
        "same_country",
    ]

    feature_cols = [
        col for col in feature_cols
        if col not in leakage_like_features
    ]

    X = scored[feature_cols]

    scored["similarity_score"] = model.predict_proba(X)[:, 1]

    if "has_garbage_entity" in scored.columns:
        scored.loc[
            scored["has_garbage_entity"] == 1,
            "similarity_score"
        ] = 0.0

    return scored


def search_top_k(
    record: dict,
    database_df: pd.DataFrame,
    model,
    top_k: int = 10,
    block_cols: list[str] | None = None,
    deduplicate_results: bool = True,
) -> pd.DataFrame:
    """
    Главная функция поиска.

    Возвращает top-K наиболее похожих записей.
    """

    if block_cols is None:
        block_cols = [
            "block_prefix_len",
            "block_first_token_len",
            "block_sorted_tokens",
        ]

    query_df = build_query_dataframe(record)

    candidates = get_candidate_records(
        query_df=query_df,
        database_df=database_df,
        block_cols=block_cols,
    )

    if candidates.empty:
        return pd.DataFrame()

    pairs = build_query_candidate_pairs(
        query_df=query_df,
        candidates=candidates,
    )

    scored = score_search_pairs(
    pairs=pairs,
    model=model,
    )

    scored = scored.sort_values(
        "similarity_score",
        ascending=False,
    )

    if deduplicate_results:
        scored = scored.drop_duplicates("name_latin_2")
    
    result = (
        scored
        .head(top_k)
        .reset_index(drop=True)
    )

    output_cols = [
        "record_id_2",
        "party_name_2",
        "name_latin_2",
        "country_2",
        "relation_kind_2",
        "relation_role_2",
        "company_public_id_2",
        "company_name_norm_2",
        "similarity_score",
    ]

    return result[output_cols]