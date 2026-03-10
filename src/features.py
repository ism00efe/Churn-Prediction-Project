import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def encode_binary_columns(df, binary_cols):
    """Evet/Hayır ve Cinsiyet gibi ikili sütunları 1 ve 0'a dönüştürür."""
    df = df.copy()
    for col in binary_cols:
        if col in df.columns:
            if col == "gender":
                df[col] = df[col].replace({"Female": 1, "Male": 0})
            else:
                df[col] = df[col].replace({"No": 0, "Yes": 1, "False": 0, "True": 1})
    return df

def encode_manual_ordinal(df):
    """Hiyerarşisi olan veya özel dönüşüm gerektiren sütunları işler."""
    df = df.copy()
    if 'Contract' in df.columns:
        df['Contract'] = df['Contract'].replace({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].replace({'No': 0, 'Yes': 1, 'No phone service': 2})
    return df

def apply_one_hot_encoding(df, ohe_cols):
    """Belirtilen sütunlara One-Hot Encoding uygular."""
    # errors='ignore' mantığı için mevcut sütunları filtrele
    valid_cols = [col for col in ohe_cols if col in df.columns]
    return pd.get_dummies(df, columns=valid_cols, drop_first=False)

def filter_by_correlation(df, target_col, threshold):
    """Hedef değişken ile korelasyonu eşik değerin altında olan sütunları siler."""
    df = df.copy()

    if target_col not in df.columns:
        return df

    target = df[target_col]

    # Hedef değişken sayısal değilse, korelasyonu hesaplayabilmek için 0/1'e zorla
    if not pd.api.types.is_numeric_dtype(target):
        target_mapped = target.replace(
            {"No": 0, "Yes": 1, "False": 0, "True": 1}
        )
        target = pd.to_numeric(target_mapped, errors="coerce")

    # Korelasyon hesabı için tüm sütunları (mümkün olduğunca) sayısala zorla
    corr_df = df.copy()
    corr_df[target_col] = target
    for col in corr_df.columns:
        if col == target_col:
            continue
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")

    # Sadece korelasyonu hesaplanabilen (NaN olmayan) sütunları al
    full_corrs = corr_df.corr()[target_col]
    corrs = full_corrs.abs().dropna()

    low_corr_cols = corrs[corrs < threshold].index.tolist()

    # Hedef değişkenin kendisini asla silme
    low_corr_cols = [col for col in low_corr_cols if col != target_col]

    df = df.drop(columns=low_corr_cols, errors="ignore")
    return df

def drop_multicollinearity(df, columns_to_drop):
    """Aralarında çok yüksek korelasyon olan ve modeli bozan sütunları siler."""
    return df.drop(columns=columns_to_drop, errors="ignore")

def optimize_data_types(df):
    """Kalan tüm verileri sayısala zorlar ve bellek optimizasyonu (downcasting) yapar."""
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].astype('float32') # Hassasiyeti koru
        else:
            # Önce sayısala zorla, çevrilemeyen metin/boşluk kalırsa 0 ile doldur
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Taşma (overflow) riskine karşı veri tiplerini küçült
            if col == 'tenure':
                df[col] = df[col].astype('int16')
            elif col not in ['MonthlyCharges', 'TotalCharges_num']:
                df[col] = df[col].astype('int8')
                
    return df

def build_features(df, config):
    """Tüm özellik mühendisliği adımlarını yöneten orkestratör fonksiyon."""
    df = df.copy()
    
    df = encode_binary_columns(df, config.BINARY_COLS)
    df = encode_manual_ordinal(df)
    df = apply_one_hot_encoding(df, config.OHE_COLS)
    
    # Hedefle ilişkisi çok düşük olanları at (0.10)
    df = filter_by_correlation(df, target_col="Churn", threshold=config.CORRELATION_THRESHOLD)
    
    # Kendi içinde yüksek korelasyonlu olanları at (TotalCharges_num)
    df = drop_multicollinearity(df, config.MULTICOLLINEARITY_DROP)

    df = optimize_data_types(df)
    
    return df

def build_preprocessor(X_train):
    """Sayısal sütunlar için scaler içeren preprocessor oluşturur."""
    numeric_features = [col for col in X_train.columns if X_train[col].nunique() > 2]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
        ],
        remainder='passthrough'
    )
    return preprocessor
