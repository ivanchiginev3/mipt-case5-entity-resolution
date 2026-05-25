"""Полный preprocessing пайплайн"""
from src.normalization import (
    advanced_normalize, remove_stopwords, 
    normalize_legal_forms, remove_legal_forms
)
from src.transliteration import transliterate_to_latin

def preprocess_dataset(df):
    """Применение всех шагов preprocessing"""
    
    print("Preprocessing...")
    df['name_advanced'] = df['party_name'].apply(advanced_normalize)
    df['name_no_stopwords'] = df['name_advanced'].apply(remove_stopwords)
    df['name_legal_norm'] = df['name_no_stopwords'].apply(normalize_legal_forms)
    df['name_no_legal'] = df['name_legal_norm'].apply(remove_legal_forms)
    df['name_latin_advanced'] = df['name_no_legal'].apply(transliterate_to_latin)
    print("✅ Done!")
    return df


def preprocessing_stats(df):
    """Статистика после preprocessing"""
    print("\nPREPROCESSING STATS:")
    for col in ['party_name', 'name_advanced', 'name_no_legal', 'name_latin_advanced']:
        if col in df.columns:
            n = df[col].nunique()
            e = (df[col].isna().sum() + (df[col] == '').sum())
            print(f"  {col}: unique={n:,}, empty={e:,}")