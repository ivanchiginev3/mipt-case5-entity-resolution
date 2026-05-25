"""
Главный pipeline
"""

from src.pipeline.cleaning import (
    remove_garbage_rows,
    add_script_column,
    remove_invalid_scripts,
    remove_empty_names,
    parse_dates
)

from src.pipeline.preprocessing import (
    preprocess_names
)


def build_dataset(df):

    print("\nSTART DATASET BUILDING\n")

    # очистка
    df = remove_garbage_rows(df)

    df = remove_empty_names(df)

    df = add_script_column(df)

    df = remove_invalid_scripts(df)

    df = parse_dates(df)

    # preprocessing
    df = preprocess_names(df)

    print("\nFINAL DATASET SIZE")
    print(len(df))

    return df