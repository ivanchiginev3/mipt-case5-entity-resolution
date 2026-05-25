"""
FastAPI app for Entity Resolution.

Endpoints:
- POST /search
- POST /check_duplicate
- POST /deduplicate_batch
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.blocking import make_block_keys
from src.pipeline.search import search_top_k
from src.features.text_features import build_pair_features, get_feature_columns
from src.pipeline.inference import run_inference_pipeline
from src.models.clustering import (
    build_duplicate_graph,
    build_clusters_from_graph,
    attach_clusters_to_records,
    add_canonical_names,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "full_dedup"
    / "final_datasets"
    / "entities.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "random_forest_matcher.joblib"
)

DUPLICATE_THRESHOLD = 0.90


app = FastAPI(
    title="ClearPic Entity Resolution API",
    description="Multilingual fuzzy matching and deduplication API",
    version="1.0.0",
)


class EntityRecord(BaseModel):
    party_name: str
    country: Optional[str] = None
    relation_kind: Optional[str] = None
    relation_role: Optional[str] = None
    company_public_id: Optional[str] = None
    company_name_norm: Optional[str] = None
    party_public_id: Optional[str] = None


class SearchRequest(BaseModel):
    record: EntityRecord
    top_k: int = Field(default=10, ge=1, le=100)


class DuplicateCheckRequest(BaseModel):
    record_1: EntityRecord
    record_2: EntityRecord


class DeduplicateBatchRequest(BaseModel):
    records: List[EntityRecord]
    score_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


@app.on_event("startup")
def load_resources():
    global model
    global search_index

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("Building search index...")
    search_index = make_block_keys(
        df.copy(),
        name_col="name_latin",
    )

    print("API is ready.")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "ClearPic Entity Resolution API",
        "endpoints": [
            "/search",
            "/check_duplicate",
            "/deduplicate_batch",
        ],
    }


@app.post("/search")
def search(request: SearchRequest):
    results = search_top_k(
        record=request.record.model_dump(),
        database_df=search_index,
        model=model,
        top_k=request.top_k,
    )

    return {
        "query": request.record.model_dump(),
        "top_k": request.top_k,
        "results": results.to_dict(orient="records"),
    }


def build_pair_for_check(record_1: Dict[str, Any], record_2: Dict[str, Any]) -> pd.DataFrame:
    from src.pipeline.search import preprocess_query_record

    r1 = preprocess_query_record(record_1)
    r2 = preprocess_query_record(record_2)

    pair = {
        "idx1": 0,
        "idx2": 1,

        "record_id_1": "record_1",
        "party_public_id_1": r1.get("party_public_id"),
        "party_name_1": r1.get("party_name"),
        "name_latin_1": r1.get("name_latin"),
        "name_no_legal_1": r1.get("name_no_legal"),
        "country_1": r1.get("country"),
        "script_1": r1.get("script"),
        "relation_kind_1": r1.get("relation_kind"),
        "relation_role_1": r1.get("relation_role"),
        "company_public_id_1": r1.get("company_public_id"),
        "company_name_norm_1": r1.get("company_name_norm"),

        "record_id_2": "record_2",
        "party_public_id_2": r2.get("party_public_id"),
        "party_name_2": r2.get("party_name"),
        "name_latin_2": r2.get("name_latin"),
        "name_no_legal_2": r2.get("name_no_legal"),
        "country_2": r2.get("country"),
        "script_2": r2.get("script"),
        "relation_kind_2": r2.get("relation_kind"),
        "relation_role_2": r2.get("relation_role"),
        "company_public_id_2": r2.get("company_public_id"),
        "company_name_norm_2": r2.get("company_name_norm"),
    }

    return pd.DataFrame([pair])


@app.post("/check_duplicate")
def check_duplicate(request: DuplicateCheckRequest):
    pair_df = build_pair_for_check(
        request.record_1.model_dump(),
        request.record_2.model_dump(),
    )

    scored = build_pair_features(pair_df)

    feature_cols = get_feature_columns()

    leakage_like_features = [
        "same_company_public_id",
        "same_country",
    ]

    feature_cols = [
        col for col in feature_cols
        if col not in leakage_like_features
    ]

    score = float(model.predict_proba(scored[feature_cols])[:, 1][0])

    if "has_garbage_entity" in scored.columns:
        if int(scored["has_garbage_entity"].iloc[0]) == 1:
            score = 0.0

    return {
        "record_1": request.record_1.model_dump(),
        "record_2": request.record_2.model_dump(),
        "duplicate_probability": score,
        "is_duplicate": score >= 0.9,
    }


@app.post("/deduplicate_batch")
def deduplicate_batch(request: DeduplicateBatchRequest):
    rows = []

    for i, record in enumerate(request.records):
        row = record.model_dump()
        row["record_id"] = f"batch_record_{i}"
        rows.append(row)

    batch_df = pd.DataFrame(rows)

    from src.pipeline.preprocessing import preprocess_names

    batch_df = preprocess_names(batch_df)

    # минимальные поля, которых может не быть
    for col in [
        "country",
        "relation_kind",
        "relation_role",
        "company_public_id",
        "company_name_norm",
        "party_public_id",
        "script",
    ]:
        if col not in batch_df.columns:
            batch_df[col] = None

    scored_pairs = run_inference_pipeline(
        df=batch_df,
        model=model,
        max_block_size=500,
        score_threshold=request.score_threshold,
    )

    duplicate_pairs = scored_pairs[
        scored_pairs["duplicate_score"] >= request.score_threshold
    ].copy()

    graph = build_duplicate_graph(
        predicted_pairs=duplicate_pairs,
        score_col="duplicate_score",
        threshold=request.score_threshold,
    )

    clusters = build_clusters_from_graph(graph)

    clustered = attach_clusters_to_records(
        df=batch_df,
        clusters=clusters,
    )

    clustered = add_canonical_names(
        clustered,
        name_col="name_latin",
    )

    return {
        "input_records": len(batch_df),
        "duplicate_pairs": len(duplicate_pairs),
        "entities": clustered[
            [
                "record_id",
                "party_name",
                "name_latin",
                "entity_id",
                "cluster_size",
                "canonical_name",
            ]
        ].to_dict(orient="records"),
    }