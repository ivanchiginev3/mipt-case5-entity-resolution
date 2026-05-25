"""
Block cleaning utilities.

Цель:
1. Найти слишком большие и слишком общие блоки.
2. Пометить stop-blocks.
3. Удалить плохие блоки из candidate generation.
"""

import pandas as pd


GENERIC_TOKENS = {
    "unknown",
    "anhayt",
    "kazmakerpoutyoun",
    "company",
    "organization",
    "kompania",
    "holding",
    "group",
    "limited",
    "trade",
    "investment",
    "capital",
    "enterprise",
    "firm",
}


def get_block_stats(
    df: pd.DataFrame,
    block_col: str,
) -> pd.DataFrame:
    """
    Считает размеры блоков.
    """

    stats = (
        df.groupby(block_col)
        .size()
        .reset_index(name="block_size")
        .sort_values("block_size", ascending=False)
        .reset_index(drop=True)
    )

    return stats


def mark_large_blocks(
    block_stats: pd.DataFrame,
    max_block_size: int = 500,
) -> pd.DataFrame:
    """
    Помечает слишком большие блоки.
    """

    block_stats = block_stats.copy()

    block_stats["is_large_block"] = (
        block_stats["block_size"] > max_block_size
    )

    return block_stats


def is_generic_block_value(value: str) -> bool:
    """
    Проверяет, является ли значение блока слишком общим.
    """

    value = str(value).lower()

    tokens = set(value.replace("_", " ").split())

    if not tokens:
        return True

    return len(tokens & GENERIC_TOKENS) > 0


def mark_generic_blocks(
    block_stats: pd.DataFrame,
    block_col: str,
) -> pd.DataFrame:
    """
    Помечает generic blocks.
    """

    block_stats = block_stats.copy()

    block_stats["is_generic_block"] = (
        block_stats[block_col]
        .apply(is_generic_block_value)
    )

    return block_stats


def build_stop_blocks(
    df: pd.DataFrame,
    block_col: str,
    max_block_size: int = 500,
) -> pd.DataFrame:
    """
    Создает таблицу stop-blocks.
    """

    stats = get_block_stats(df, block_col)

    stats = mark_large_blocks(
        stats,
        max_block_size=max_block_size,
    )

    stats = mark_generic_blocks(
        stats,
        block_col=block_col,
    )

    stats["is_stop_block"] = (
        stats["is_large_block"]
        |
        stats["is_generic_block"]
    )

    return stats


def filter_stop_blocks(
    df: pd.DataFrame,
    block_col: str,
    stop_blocks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Удаляет записи, попавшие в stop-blocks.
    """

    bad_values = set(
        stop_blocks[
            stop_blocks["is_stop_block"]
        ][block_col]
    )

    result = df[
        ~df[block_col].isin(bad_values)
    ].copy()

    return result


def summarize_stop_blocks(
    stop_blocks: pd.DataFrame,
) -> None:
    """
    Печатает отчет по stop-blocks.
    """

    print("=" * 60)
    print("STOP BLOCK SUMMARY")
    print("=" * 60)

    print(f"Total blocks: {len(stop_blocks):,}")

    print("\nStop block counts:")
    print(stop_blocks["is_stop_block"].value_counts())

    print("\nLarge blocks:")
    print(stop_blocks["is_large_block"].sum())

    print("\nGeneric blocks:")
    print(stop_blocks["is_generic_block"].sum())

    print("\nTop stop blocks:")
    display(
        stop_blocks[
            stop_blocks["is_stop_block"]
        ]
        .sort_values("block_size", ascending=False)
        .head(20)
    )