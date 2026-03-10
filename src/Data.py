import pandas as pd
import numpy as np

def drop_unnecessary_columns(df, columns):
    """Belirtilen sütunları veri setinden kaldırır."""
    return df.drop(columns=columns, errors="ignore")

def handle_missing_and_invalid_values(df):
    """TotalCharges'ı sayısala çevirir, null ve tenure=0 olanları siler."""
    # TotalCharges'ı sayısal yap, hata verenleri (boşluklar) NaN yap
    df['TotalCharges_num'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Null olanları sil
    df = df.dropna(subset=['TotalCharges_num'])
    
    # Tenure 0 olanları sil
    df = df[df['tenure'] != 0]
    
    return df



def clean_data(df, config_cols_to_drop):
    """Tüm veri temizleme adımlarını yöneten orkestratör fonksiyon."""
    df = df.copy()
    
    df = handle_missing_and_invalid_values(df)
    df = drop_unnecessary_columns(df, config_cols_to_drop)
    
    return df