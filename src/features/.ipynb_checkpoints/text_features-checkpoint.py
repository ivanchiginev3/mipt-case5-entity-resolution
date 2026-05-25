"""
Feature engineering for pairwise Entity Resolution.
"""

import re
import pandas as pd
from rapidfuzz import fuzz
from src.pipeline.garbage import is_garbage_entity


LEGAL_FORMS = {
    "llc", "ltd", "inc", "corp",
    "ooo", "zao", "oao", "pao",
    "spy", "pby", "too", "ao",
}


def tokenize(text: str) -> list[str]:
    if pd.isna(text):
        return []

    return str(text).lower().split()


def token_jaccard(text1: str, text2: str) -> float:
    t1 = set(tokenize(text1))
    t2 = set(tokenize(text2))

    if not t1 and not t2:
        return 1.0

    if not t1 or not t2:
        return 0.0

    return len(t1 & t2) / len(t1 | t2)


def common_token_count(text1: str, text2: str) -> int:
    return len(set(tokenize(text1)) & set(tokenize(text2)))


def token_count(text: str) -> int:
    return len(tokenize(text))


def char_length(text: str) -> int:
    if pd.isna(text):
        return 0

    return len(str(text))


def extract_legal_forms(text: str) -> set[str]:
    tokens = set(tokenize(text))
    return tokens & LEGAL_FORMS


def remove_legal_forms_from_text(text: str) -> str:
    tokens = tokenize(text)

    tokens = [
        t for t in tokens
        if t not in LEGAL_FORMS
    ]

    return " ".join(tokens)


def has_legal_form(text: str) -> int:
    return int(len(extract_legal_forms(text)) > 0)


def same_legal_form(text1: str, text2: str) -> int:
    lf1 = extract_legal_forms(text1)
    lf2 = extract_legal_forms(text2)

    if not lf1 and not lf2:
        return 1

    return int(len(lf1 & lf2) > 0)


def safe_equal(a, b) -> int:
    if pd.isna(a) or pd.isna(b):
        return 0

    return int(a == b)


def add_pairwise_text_features(
    pairs: pd.DataFrame,
    name_col_1: str = "name_latin_1",
    name_col_2: str = "name_latin_2",
) -> pd.DataFrame:
    """
    Добавляет строковые признаки.
    """

    pairs = pairs.copy()

    name1 = pairs[name_col_1].fillna("").astype(str)
    name2 = pairs[name_col_2].fillna("").astype(str)

    pairs["fuzz_ratio"] = [
        fuzz.ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    pairs["partial_ratio"] = [
        fuzz.partial_ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    pairs["token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    pairs["token_set_ratio"] = [
        fuzz.token_set_ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    pairs["exact_match"] = (
        name1 == name2
    ).astype(int)

    pairs["token_jaccard"] = [
        token_jaccard(a, b)
        for a, b in zip(name1, name2)
    ]

    pairs["common_tokens"] = [
        common_token_count(a, b)
        for a, b in zip(name1, name2)
    ]

    pairs["len_1"] = [
        char_length(a)
        for a in name1
    ]

    pairs["len_2"] = [
        char_length(b)
        for b in name2
    ]

    pairs["length_diff"] = (
        pairs["len_1"] - pairs["len_2"]
    ).abs()

    pairs["token_count_1"] = [
        token_count(a)
        for a in name1
    ]

    pairs["token_count_2"] = [
        token_count(b)
        for b in name2
    ]

    pairs["token_count_diff"] = (
        pairs["token_count_1"] - pairs["token_count_2"]
    ).abs()

    return pairs


def add_legal_form_features(
    pairs: pd.DataFrame,
    name_col_1: str = "name_latin_1",
    name_col_2: str = "name_latin_2",
) -> pd.DataFrame:
    """
    Добавляет признаки юридических форм.
    """

    pairs = pairs.copy()

    name1 = pairs[name_col_1].fillna("").astype(str)
    name2 = pairs[name_col_2].fillna("").astype(str)

    pairs["has_legal_form_1"] = [
        has_legal_form(a)
        for a in name1
    ]

    pairs["has_legal_form_2"] = [
        has_legal_form(b)
        for b in name2
    ]

    pairs["same_legal_form"] = [
        same_legal_form(a, b)
        for a, b in zip(name1, name2)
    ]

    core1 = [
        remove_legal_forms_from_text(a)
        for a in name1
    ]

    core2 = [
        remove_legal_forms_from_text(b)
        for b in name2
    ]

    pairs["core_fuzz_ratio"] = [
        fuzz.ratio(a, b)
        for a, b in zip(core1, core2)
    ]

    pairs["core_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b)
        for a, b in zip(core1, core2)
    ]

    pairs["core_token_set_ratio"] = [
        fuzz.token_set_ratio(a, b)
        for a, b in zip(core1, core2)
    ]

    pairs["core_exact_match"] = [
        int(a == b)
        for a, b in zip(core1, core2)
    ]

    return pairs


def add_metadata_features(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет признаки по метаданным.
    """

    pairs = pairs.copy()

    pairs["same_country"] = [
        safe_equal(a, b)
        for a, b in zip(
            pairs.get("country_1"),
            pairs.get("country_2"),
        )
    ]

    pairs["same_script"] = [
        safe_equal(a, b)
        for a, b in zip(
            pairs.get("script_1"),
            pairs.get("script_2"),
        )
    ]

    pairs["same_relation_kind"] = [
        safe_equal(a, b)
        for a, b in zip(
            pairs.get("relation_kind_1"),
            pairs.get("relation_kind_2"),
        )
    ]

    pairs["same_relation_role"] = [
        safe_equal(a, b)
        for a, b in zip(
            pairs.get("relation_role_1"),
            pairs.get("relation_role_2"),
        )
    ]

    pairs["same_company_public_id"] = [
        safe_equal(a, b)
        for a, b in zip(
            pairs.get("company_public_id_1"),
            pairs.get("company_public_id_2"),
        )
    ]
    pairs["is_garbage_1"] = [
    is_garbage_entity(x)
    for x in pairs["party_name_1"]
    ]

    pairs["is_garbage_2"] = [
        is_garbage_entity(x)
        for x in pairs["party_name_2"]
    ]

    pairs["has_garbage_entity"] = (
        (pairs["is_garbage_1"] == 1)
        |
        (pairs["is_garbage_2"] == 1)
    ).astype(int)

    return pairs


def build_pair_features(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Полный pipeline построения признаков.
    """

    pairs = add_pairwise_text_features(pairs)
    pairs = add_legal_form_features(pairs)
    pairs = add_metadata_features(pairs)

    return pairs


def get_feature_columns() -> list[str]:
    """
    Список признаков для модели.
    """

    return [
        "fuzz_ratio",
        "partial_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "exact_match",
        "token_jaccard",
        "common_tokens",
        "len_1",
        "len_2",
        "length_diff",
        "token_count_1",
        "token_count_2",
        "token_count_diff",
        "has_legal_form_1",
        "has_legal_form_2",
        "same_legal_form",
        "core_fuzz_ratio",
        "core_token_sort_ratio",
        "core_token_set_ratio",
        "core_exact_match",
        "same_script",
        "same_relation_kind",
        "same_relation_role",
        "is_garbage_1",
        "is_garbage_2",
        "has_garbage_entity",
    ]