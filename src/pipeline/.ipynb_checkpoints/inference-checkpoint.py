"""
Inference pipeline for Entity Resolution.

Pipeline:
records
→ blocking
→ candidate pairs
→ feature generation
→ model scoring
"""

import pandas as pd
import joblib

from src.features.blocking import (
    make_block_keys,
    generate_candidate_pairs_from_block,
)

from src.features.text_features import (
    build_pair_features,
    get_feature_columns,
)


def deduplicate_candidate_pairs(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    candidates = candidates.copy()

    candidates["left"] = candidates[["idx1", "idx2"]].min(axis=1)
    candidates["right"] = candidates[["idx1", "idx2"]].max(axis=1)

    candidates["pair_key"] = (
        candidates["left"].astype(str)
        + "_"
        + candidates["right"].astype(str)
    )

    candidates = (
        candidates
        .drop_duplicates("pair_key")
        .reset_index(drop=True)
    )

    return candidates


def generate_multipass_candidates(
    df: pd.DataFrame,
    block_cols: list[str],
    max_block_size: int = 500,
) -> pd.DataFrame:
    candidate_frames = []

    for block_col in block_cols:
        candidates = generate_candidate_pairs_from_block(
            df=df,
            block_col=block_col,
            max_block_size=max_block_size,
        )

        candidate_frames.append(candidates)

    if not candidate_frames:
        return pd.DataFrame(columns=["idx1", "idx2"])

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    candidates = deduplicate_candidate_pairs(candidates)

    return candidates


def attach_records_to_candidates(
    candidates: pd.DataFrame,
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    pairs = candidates.copy()

    left = df[columns].copy()
    right = df[columns].copy()

    left.columns = [f"{col}_1" for col in columns]
    right.columns = [f"{col}_2" for col in columns]

    pairs = pairs.join(left, on="idx1")
    pairs = pairs.join(right, on="idx2")

    return pairs


def score_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> pd.DataFrame:
    scored = build_pair_features(candidate_pairs)

    X = scored[feature_cols]

    scored["duplicate_score"] = model.predict_proba(X)[:, 1]
    scored["duplicate_prediction"] = model.predict(X)

    return scored


def run_inference_pipeline(
    df: pd.DataFrame,
    model,
    block_cols: list[str] | None = None,
    max_block_size: int = 500,
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    if block_cols is None:
        block_cols = [
            "block_prefix_len",
            "block_first_token_len",
            "block_sorted_tokens",
        ]

    df = make_block_keys(
        df,
        name_col="name_latin",
    )

    candidates = generate_multipass_candidates(
        df=df,
        block_cols=block_cols,
        max_block_size=max_block_size,
    )

    columns_to_attach = [
        "record_id",
        "party_public_id",
        "party_name",
        "name_latin",
        "name_no_legal",
        "country",
        "script",
        "relation_kind",
        "relation_role",
        "company_public_id",
        "company_name_norm",
    ]

    candidate_pairs = attach_records_to_candidates(
        candidates=candidates,
        df=df,
        columns=columns_to_attach,
    )

    feature_cols = get_feature_columns()

    leakage_like_features = [
        "same_company_public_id",
        "same_country",
    ]

    feature_cols = [
        col for col in feature_cols
        if col not in leakage_like_features
    ]

    scored = score_candidate_pairs(
        candidate_pairs=candidate_pairs,
        model=model,
        feature_cols=feature_cols,
    )

    scored["duplicate_prediction"] = (
        scored["duplicate_score"] >= score_threshold
    ).astype(int)

    return scored