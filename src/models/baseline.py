"""
Baseline matching methods for Entity Resolution.
"""

import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


def add_baseline_scores(
    pairs: pd.DataFrame,
    name_col_1: str = "name_latin_1",
    name_col_2: str = "name_latin_2",
) -> pd.DataFrame:
    """
    Добавляет простые baseline scores.
    """

    pairs = pairs.copy()

    name1 = pairs[name_col_1].fillna("").astype(str)
    name2 = pairs[name_col_2].fillna("").astype(str)

    pairs["exact_match_score"] = (
        name1 == name2
    ).astype(int)

    pairs["fuzz_ratio"] = [
        fuzz.ratio(a, b)
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

    return pairs


def predict_by_threshold(
    pairs: pd.DataFrame,
    score_col: str,
    threshold: float,
) -> pd.Series:
    """
    Делает предсказание по порогу.
    """

    return (
        pairs[score_col] >= threshold
    ).astype(int)


def evaluate_predictions(
    y_true,
    y_pred,
    y_score=None,
) -> dict:
    """
    Считает основные метрики.
    """

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            result["roc_auc"] = None

    return result


def evaluate_baselines(
    pairs: pd.DataFrame,
    thresholds: dict,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Оценивает несколько baseline-методов.
    """

    rows = []

    y_true = pairs[label_col]

    for score_col, threshold_values in thresholds.items():
        for threshold in threshold_values:
            y_pred = predict_by_threshold(
                pairs,
                score_col=score_col,
                threshold=threshold,
            )

            y_score = pairs[score_col]

            metrics = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
            )

            rows.append(
                {
                    "method": score_col,
                    "threshold": threshold,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def get_confusion_matrix_df(
    y_true,
    y_pred,
) -> pd.DataFrame:
    """
    Возвращает confusion matrix как DataFrame.
    """

    cm = confusion_matrix(y_true, y_pred)

    return pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )