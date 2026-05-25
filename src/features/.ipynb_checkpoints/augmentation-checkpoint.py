"""
Data augmentation for Entity Resolution train pairs.

Важно:
аугментацию применяем только к train_pairs.
valid/test не изменяем.
"""

import random
import re
import pandas as pd


LEGAL_FORMS_LATIN = [
    "llc",
    "ltd",
    "inc",
    "corp",
    "ooo",
    "zao",
    "oao",
    "pao",
    "spy",
    "pby",
]

COMPANY_SUFFIXES = [
    "group",
    "holding",
    "plus",
    "trade",
    "capital",
    "investment",
]


def drop_random_token(text: str) -> str:
    tokens = str(text).split()

    if len(tokens) <= 1:
        return str(text)

    idx = random.randrange(len(tokens))
    return " ".join(tokens[:idx] + tokens[idx + 1:])


def swap_tokens(text: str) -> str:
    tokens = str(text).split()

    if len(tokens) <= 1:
        return str(text)

    i, j = random.sample(range(len(tokens)), 2)
    tokens[i], tokens[j] = tokens[j], tokens[i]

    return " ".join(tokens)


def add_typo(text: str) -> str:
    text = str(text)

    if len(text) <= 3:
        return text

    idx = random.randrange(len(text))

    # удаляем один символ
    return text[:idx] + text[idx + 1:]


def remove_legal_form(text: str) -> str:
    tokens = str(text).split()

    tokens = [
        t for t in tokens
        if t not in LEGAL_FORMS_LATIN
    ]

    return " ".join(tokens)


def add_legal_form(text: str) -> str:
    text = str(text).strip()

    if not text:
        return text

    legal_form = random.choice(LEGAL_FORMS_LATIN)

    if random.random() < 0.5:
        return f"{text} {legal_form}"

    return f"{legal_form} {text}"


def change_case_noise(text: str) -> str:
    text = str(text)

    if random.random() < 0.5:
        return text.upper()

    return text.title()


def remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def augment_positive_name(text: str) -> str:
    """
    Создает вариант того же имени.
    """

    operations = [
        drop_random_token,
        swap_tokens,
        add_typo,
        remove_legal_form,
        add_legal_form,
        change_case_noise,
    ]

    operation = random.choice(operations)

    return remove_extra_spaces(operation(text))


def augment_company_like_negative(text: str) -> str:
    """
    Создает похожее, но потенциально другое название компании.
    """

    text = str(text).strip()

    suffix = random.choice(COMPANY_SUFFIXES)

    if random.random() < 0.5:
        return f"{text} {suffix}"

    return f"{suffix} {text}"


def augment_train_pairs(
    train_pairs: pd.DataFrame,
    n_positive_aug: int = 30_000,
    n_hard_negative_aug: int = 30_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Добавляет искусственные пары к train.

    Positive augmentation:
    name_1 остается исходным,
    name_2 искусственно искажается, label=1.

    Hard negative augmentation:
    создается похожее, но отличающееся имя, label=0.
    """

    random.seed(random_state)

    rows = []

    positive_source = train_pairs[
        train_pairs["label"] == 1
    ].copy()

    hard_negative_source = train_pairs[
        train_pairs["pair_type"] == "hard_negative_similar_name"
    ].copy()

    if len(positive_source) > 0:
        sampled_pos = positive_source.sample(
            min(n_positive_aug, len(positive_source)),
            random_state=random_state,
            replace=len(positive_source) < n_positive_aug,
        )

        for _, row in sampled_pos.iterrows():
            new_row = row.copy()

            new_row["name_latin_2"] = augment_positive_name(
                row["name_latin_2"]
            )

            new_row["party_name_2"] = new_row["name_latin_2"]

            new_row["pair_type"] = "augmented_positive"

            rows.append(new_row)

    if len(hard_negative_source) > 0:
        sampled_neg = hard_negative_source.sample(
            min(n_hard_negative_aug, len(hard_negative_source)),
            random_state=random_state + 1,
            replace=len(hard_negative_source) < n_hard_negative_aug,
        )

        for _, row in sampled_neg.iterrows():
            new_row = row.copy()

            base_name = row["name_latin_1"]

            new_row["name_latin_2"] = augment_company_like_negative(
                base_name
            )

            new_row["party_name_2"] = new_row["name_latin_2"]

            new_row["label"] = 0
            new_row["pair_type"] = "augmented_hard_negative"

            rows.append(new_row)

    if not rows:
        return train_pairs.copy()

    augmented_rows = pd.DataFrame(rows)

    result = pd.concat(
        [train_pairs, augmented_rows],
        ignore_index=True,
    )

    result = result.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    return result

TYPO_CHAR_MAP = {
    "ҳ": "h",
    "ҷ": "ch",
    "қ": "k",
    "ғ": "g",
    "ӯ": "u",
    "ӣ": "i",
    "ə": "e",
    "ğ": "g",
    "ş": "sh",
    "ç": "ch",
}


def delete_random_char(text: str) -> str:
    text = str(text)

    if len(text) <= 3:
        return text

    idx = random.randrange(len(text))
    return text[:idx] + text[idx + 1:]


def swap_adjacent_chars(text: str) -> str:
    text = str(text)

    if len(text) <= 3:
        return text

    idx = random.randrange(len(text) - 1)

    return (
        text[:idx]
        + text[idx + 1]
        + text[idx]
        + text[idx + 2:]
    )


def replace_national_char(text: str) -> str:
    text = str(text)

    candidates = [
        ch for ch in text
        if ch.lower() in TYPO_CHAR_MAP
    ]

    if not candidates:
        return text

    ch = random.choice(candidates)

    return text.replace(
        ch,
        TYPO_CHAR_MAP.get(ch.lower(), ch),
        1,
    )


def remove_patronymic(text: str) -> str:
    tokens = str(text).split()

    if len(tokens) < 3:
        return text

    return " ".join(tokens[:2])


def realistic_positive_augmentation(text: str) -> str:
    """
    Более реалистичная positive-аугментация:
    маленькие опечатки, перестановки, удаление отчества.
    """

    operations = [
        delete_random_char,
        swap_adjacent_chars,
        replace_national_char,
        remove_patronymic,
        swap_tokens,
        remove_legal_form,
        add_legal_form,
    ]

    operation = random.choice(operations)

    return remove_extra_spaces(operation(text))