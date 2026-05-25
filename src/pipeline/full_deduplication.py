"""
Parallel full deduplication pipeline.

Country-wise + block-wise + chunk-wise processing.

Main idea:
- do not keep all candidate pairs in RAM;
- process chunks in parallel;
- save duplicate edges per chunk;
- allow resume after crash.
"""

from pathlib import Path
from itertools import combinations
import gc

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from src.features.blocking import make_block_keys
from src.features.text_features import build_pair_features, get_feature_columns
from src.models.clustering import (
    build_duplicate_graph,
    build_clusters_from_graph,
    attach_clusters_to_records,
    add_canonical_names,
)
from src.models.canonicalization import (
    build_canonical_entities,
    add_entity_quality_flags,
)


def get_model_feature_cols():
    feature_cols = get_feature_columns()

    leakage_like_features = [
        "same_company_public_id",
        "same_country",
    ]

    return [
        col for col in feature_cols
        if col not in leakage_like_features
    ]


def deduplicate_edges(edges):
    if edges.empty:
        return edges

    edges["left"] = edges["idx1"].where(
        edges["idx1"] <= edges["idx2"],
        edges["idx2"]
    )

    edges["right"] = edges["idx2"].where(
        edges["idx1"] <= edges["idx2"],
        edges["idx1"]
    )

    edges = edges[["left", "right", "duplicate_score"]]

    edges = edges.rename(
        columns={
            "left": "idx1",
            "right": "idx2"
        }
    )

    edges["idx1"] = edges["idx1"].astype("int32")
    edges["idx2"] = edges["idx2"].astype("int32")
    edges["duplicate_score"] = edges["duplicate_score"].astype("float32")

    edges = edges.sort_values(
        ["idx1", "idx2", "duplicate_score"],
        ascending=[True, True, False]
    )

    edges = edges.drop_duplicates(
        subset=["idx1", "idx2"],
        keep="first"
    )

    return edges.reset_index(drop=True)


def attach_records_to_pairs(
    pairs_df: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
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

    existing_columns = [
        col for col in columns
        if col in df.columns
    ]

    left = df[existing_columns].copy()
    right = df[existing_columns].copy()

    left.columns = [f"{col}_1" for col in existing_columns]
    right.columns = [f"{col}_2" for col in existing_columns]

    result = pairs_df.copy()
    result = result.join(left, on="idx1")
    result = result.join(right, on="idx2")

    return result


def score_pairs_chunk_to_file(
    chunk_id: str,
    pairs_df: pd.DataFrame,
    df_country: pd.DataFrame,
    model,
    feature_cols: list[str],
    score_threshold: float,
    output_dir: str | Path,
) -> str:
    """
    Scores one chunk and saves duplicate edges to parquet.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = output_dir / f"edges_chunk_{chunk_id}.parquet"

    if chunk_path.exists():
        return str(chunk_path)

    if pairs_df.empty:
        pd.DataFrame().to_parquet(chunk_path, index=False)
        return str(chunk_path)

    pair_records = attach_records_to_pairs(
        pairs_df=pairs_df,
        df=df_country,
    )

    features = build_pair_features(pair_records)

    X = features[feature_cols]

    features["duplicate_score"] = model.predict_proba(X)[:, 1]

    if "has_garbage_entity" in features.columns:
        features.loc[
            features["has_garbage_entity"] == 1,
            "duplicate_score",
        ] = 0.0

    edges = features[
        features["duplicate_score"] >= score_threshold
    ].copy()

    keep_cols = [
        "idx1",
        "idx2",
        "duplicate_score",
        "party_name_1",
        "party_name_2",
        "name_latin_1",
        "name_latin_2",
        "country_1",
        "country_2",
        "fuzz_ratio",
        "token_set_ratio",
        "core_token_set_ratio",
        "token_jaccard",
        "length_diff",
        "exact_match",
        "core_exact_match",
        "has_garbage_entity",
    ]

    keep_cols = [
        col for col in keep_cols
        if col in edges.columns
    ]

    edges = edges[keep_cols]

    edges.to_parquet(
        chunk_path,
        index=False,
    )

    del pair_records
    del features
    del edges
    gc.collect()

    return str(chunk_path)


def generate_pair_chunks_for_block(
    group: pd.DataFrame,
    pair_chunk_size: int,
):
    """
    Generator of pair chunks for one block.
    Does not store all pairs permanently.
    """

    indices = group.index.tolist()

    buffer = []

    for idx1, idx2 in combinations(indices, 2):
        buffer.append((idx1, idx2))

        if len(buffer) >= pair_chunk_size:
            yield pd.DataFrame(
                buffer,
                columns=["idx1", "idx2"],
            )
            buffer = []

    if buffer:
        yield pd.DataFrame(
            buffer,
            columns=["idx1", "idx2"],
        )


def run_country_deduplication_parallel(
    df_country: pd.DataFrame,
    model,
    country: str,
    output_path: str | Path,
    temp_dir: str | Path,
    block_cols: list[str] | None = None,
    max_block_size: int = 500,
    pair_chunk_size: int = 200_000,
    score_threshold: float = 0.90,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """
    Full deduplication for one country using parallel chunk scoring.
    """

    if block_cols is None:
        block_cols = [
            "block_prefix_len",
            "block_first_token_len",
            "block_sorted_tokens",
        ]

    output_path = Path(output_path)
    temp_dir = Path(temp_dir) / country

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Final edge file already exists: {output_path}")
        return pd.read_parquet(output_path)

    df_country = make_block_keys(
        df_country.copy(),
        name_col="name_latin",
    )

    feature_cols = get_model_feature_cols()

    print("=" * 80)
    print(f"COUNTRY: {country}")
    print(f"Rows: {len(df_country):,}")
    print(f"n_jobs: {n_jobs}")
    print(f"pair_chunk_size: {pair_chunk_size:,}")
    print("=" * 80)

    chunk_tasks = []
    global_buffer = []
    global_chunk_num = 0
    
    for block_col in block_cols:
        print(f"Preparing chunks for block_col={block_col}")
    
        grouped = df_country.groupby(block_col)
    
        for block_value, group in tqdm(
            grouped,
            total=df_country[block_col].nunique(),
            desc=f"Prepare {country}/{block_col}",
        ):
            block_size = len(group)
    
            if block_size < 2:
                continue
    
            if block_size > max_block_size:
                continue
    
            indices = group.index.tolist()
    
            for idx1, idx2 in combinations(indices, 2):
                global_buffer.append((idx1, idx2))
    
                if len(global_buffer) >= pair_chunk_size:
                    pairs_df = pd.DataFrame(
                        global_buffer,
                        columns=["idx1", "idx2"],
                    )
    
                    chunk_id = f"{country}_chunk_{global_chunk_num:06d}"
    
                    chunk_tasks.append(
                        (chunk_id, pairs_df)
                    )
    
                    global_buffer = []
                    global_chunk_num += 1

    # остаток
    if global_buffer:
        pairs_df = pd.DataFrame(
            global_buffer,
            columns=["idx1", "idx2"],
        )
    
        chunk_id = f"{country}_chunk_{global_chunk_num:06d}"
    
        chunk_tasks.append(
            (chunk_id, pairs_df)
        )

    print(f"Total chunks to score: {len(chunk_tasks):,}")

    chunk_paths = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
    )(
        delayed(score_pairs_chunk_to_file)(
            chunk_id=chunk_id,
            pairs_df=pairs_df,
            df_country=df_country,
            model=model,
            feature_cols=feature_cols,
            score_threshold=score_threshold,
            output_dir=temp_dir,
        )
        for chunk_id, pairs_df in chunk_tasks
    )

    edge_frames = []

    for path in tqdm(chunk_paths, desc=f"Loading chunks {country}"):
        path = Path(path)

        if not path.exists():
            continue

        try:
            edges = pd.read_parquet(path)
        except Exception:
            continue

        if not edges.empty:
            edge_frames.append(edges)

    if edge_frames:
        result_edges = pd.concat(
            edge_frames,
            ignore_index=True,
        )
        result_edges = deduplicate_edges(result_edges)
    else:
        result_edges = pd.DataFrame(
            columns=["idx1", "idx2", "duplicate_score"]
        )

    result_edges.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved final country edges: {output_path}")
    print(f"Edges found: {len(result_edges):,}")

    return result_edges


def build_full_clusters_from_edges(
    df: pd.DataFrame,
    edges_paths: list[str | Path],
    output_dedup_path: str | Path,
    output_entities_path: str | Path,
    threshold: float = 0.90,
):
    edge_frames = []

    for path in tqdm(edges_paths, desc="Loading edge files"):
        path = Path(path)

        if not path.exists():
            continue

        edges = pd.read_parquet(path)

        if not edges.empty:
            edge_frames.append(edges)

    if edge_frames:
        all_edges = pd.concat(
            edge_frames,
            ignore_index=True,
        )
        all_edges = deduplicate_edges(all_edges)
    else:
        all_edges = pd.DataFrame(
            columns=["idx1", "idx2", "duplicate_score"]
        )

    print(f"Total duplicate edges: {len(all_edges):,}")

    graph = build_duplicate_graph(
        predicted_pairs=all_edges,
        score_col="duplicate_score",
        threshold=threshold,
    )

    clusters = build_clusters_from_graph(graph)

    deduplicated = attach_clusters_to_records(
        df=df,
        clusters=clusters,
    )

    deduplicated = add_canonical_names(
        deduplicated,
        name_col="name_latin",
    )

    entities = build_canonical_entities(deduplicated)

    entities = add_entity_quality_flags(
        entities,
        large_cluster_threshold=20,
        max_unique_names=5,
        max_unique_countries=3,
    )

    output_dedup_path = Path(output_dedup_path)
    output_entities_path = Path(output_entities_path)

    output_dedup_path.parent.mkdir(parents=True, exist_ok=True)
    output_entities_path.parent.mkdir(parents=True, exist_ok=True)

    deduplicated.to_parquet(
        output_dedup_path,
        index=False,
    )

    entities.to_parquet(
        output_entities_path,
        index=False,
    )

    print("Full clustering done.")
    print(f"Deduplicated records: {len(deduplicated):,}")
    print(f"Canonical entities: {len(entities):,}")

    return deduplicated, entities