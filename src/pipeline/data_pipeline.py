"""Data Pipeline — очистка и разделение данных"""
import pandas as pd
from src.utils import detect_script
from src.normalization import basic_clean

def clean_dataset(df):
    """Очистка датасета и разделение train/search"""
    
    print(f"Исходный размер: {len(df):,}")
    
    # 1. Удалить технический мусор
    mask = df['party_name'].str.lower().str.contains('замените на', na=False)
    print(f"Удалено 'замените на': {mask.sum():,}")
    df = df[~mask]
    
    # 2. Удалить unknown (мусор)
    df['script'] = df['party_name'].apply(detect_script)
    mask = df['script'] == 'unknown'
    print(f"Удалено unknown: {mask.sum():,}")
    df = df[~mask]
    
    # 3. Исправить даты
    df['date'] = pd.to_datetime(df['source_snapshot_date'], errors='coerce')
    tjk_mask = (df['country'] == 'tjk') & (df['date'].isna())
    if tjk_mask.any():
        df.loc[tjk_mask, 'date'] = pd.to_datetime(
            df.loc[tjk_mask, 'source_snapshot_date'].astype(float), 
            unit='s', errors='coerce'
        )
    
    # 4. Базовая нормализация
    df['party_name_clean'] = df['party_name'].apply(basic_clean)
    
    # 5. Разделение
    train = df[df['party_public_id'].notna()].copy()
    search = df[df['party_public_id'].isna()].copy()
    
    print(f"Итоговый размер: {len(df):,}")
    print(f"Train: {len(train):,} | Search: {len(search):,}")
    
    return df, train, search